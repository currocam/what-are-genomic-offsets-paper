import numpy as np
import msprime
import pandas as pd
import genomic_offsets as go
import statsmodels.api as sm
import tskit
import pyslim
import sys


def genotype_matrix(ts: tskit.TreeSequence) -> np.ndarray:
    # The genotype_matrix produces one column per haplotype
    genos = ts.genotype_matrix()
    Y = np.zeros((genos.shape[1] // 2, genos.shape[0]), dtype=np.int32)
    for indi in range(Y.shape[0]):
        Y[indi, :] = np.sum(genos[:, [indi * 2, indi * 2 + 1]], axis=1)
    return Y


def allele_frequencies(ts: tskit.TreeSequence, sample_sets=None) -> np.ndarray:
    if sample_sets is None:
        sample_sets = [ts.samples()]
    n = np.array([len(x) for x in sample_sets])

    def f(x):
        return x / n

    return ts.sample_count_stat(
        sample_sets,
        f,
        len(sample_sets),
        windows="sites",
        polarised=True,
        mode="site",
        strict=False,
        span_normalise=False,
    )


def sample_sites(
    rng: np.random.RandomState,
    ts: tskit.TreeSequence,
    num_sites: int,
    num_individuals: int,
    grid_size: int = 20,
):
    """
    Choose sampling points and sample individuals.
    """
    locs = ts.individual_locations  # Nx2 or Nx3 array of individual coordinates
    if locs.shape[0] < num_sites * num_individuals:
        raise ValueError("Not enough individuals to sample.")
    num_pixels = 200
    pixel_size = np.sqrt(grid_size * grid_size / num_pixels)  # Define pixel size
    sampled_nodes, sampled_individuals = [], []
    # Generate pixel centers
    pixel_centers = [
        (x * pixel_size + pixel_size / 2, y * pixel_size + pixel_size / 2)
        for x in range(int(np.sqrt(num_pixels)))
        for y in range(int(np.sqrt(num_pixels)))
    ]
    # Randomly shuffle indexes
    sites = np.arange(len(pixel_centers))
    rng.shuffle(sites)
    selected_sites = []
    for pixel_idx in sites:
        center_x, center_y = pixel_centers[pixel_idx]
        # Find individuals within the pixel
        in_pixel = np.where(
            (locs[:, 0] >= center_x - pixel_size / 2)
            & (locs[:, 0] < center_x + pixel_size / 2)
            & (locs[:, 1] >= center_y - pixel_size / 2)
            & (locs[:, 1] < center_y + pixel_size / 2)
        )[0]
        print(in_pixel)
        if len(in_pixel) < num_individuals:
            continue
        selected_sites.append(pixel_idx)
        # Randomly sample individuals from the pixel
        sampled_inds = rng.choice(in_pixel, size=num_individuals, replace=False)
        sampled_individuals.append(sampled_inds)
        # Extract corresponding nodes
        site_nodes = []
        for i in sampled_inds:
            site_nodes.extend(ts.individual(i).nodes)
        sampled_nodes.append(site_nodes)
        if len(sampled_nodes) == num_sites:
            break
    return sampled_nodes, sampled_individuals, np.array(pixel_centers)[selected_sites]


def simulate_dataset(ts, seed, noise):
    """
    Simulate a dataset by "sampling" individuals and pre-processing tree sequence into
    the required matrixes.
    """
    # Get metadata we manually introduced in the SLiM simulation
    metadata_df = pd.DataFrame(ts.metadata["SLiM"]["user_metadata"])
    # Get the causal environmental data
    env_initial = np.array(metadata_df[["ENV1", "ENV2"]]).reshape(
        metadata_df.shape[0], 2
    )
    env_altered = np.array(metadata_df[["ENV_ALTERED1", "ENV_ALTERED2"]]).reshape(
        metadata_df.shape[0], 2
    )
    if noise:
        # Add 5 random columns (-1, 1)
        random_noise_initial = np.random.uniform(-1, 1, (metadata_df.shape[0], 5))
        random_noise_altered = np.random.uniform(-1, 1, (metadata_df.shape[0], 5))
        env_initial = np.hstack([env_initial, random_noise_initial])
        env_altered = np.hstack([env_altered, random_noise_altered])

    # Get shifted fitness after environmental change
    fitness_altered = np.array(metadata_df["FITNESS_ALTERED"]).reshape(
        metadata_df.shape[0]
    )
    # Add neutral mutations if empirical
    if noise:
        next_id = pyslim.next_slim_mutation_id(ts)
        ts = msprime.sim_mutations(
            ts,
            rate=1e-9,
            model=msprime.SLiMMutationModel(type=0, next_id=next_id),
            keep=True,
        )
        print("nSNPS", ts.num_sites)
        allele_freqs = allele_frequencies(ts)
        ts = ts.delete_sites(
            np.where(np.bitwise_or(allele_freqs < 0.01, allele_freqs > 0.99))
        )
        print("nSNPS (after filtering)", ts.num_sites)
    # Create a random dataset
    rng = np.random.RandomState(seed)
    num_sites = 10
    num_individuals = 10
    sampled_nodes, sampled_individuals, sampled_locations = sample_sites(
        rng, ts, num_sites=num_sites, num_individuals=num_individuals
    )
    # Get env-data
    # Compute mean site environmental values
    compute_mean_site = lambda mat: np.array(
        [
            np.mean(
                mat[[ts.node(i).individual for i in sampled_nodes[site]], :], axis=0
            )
            for site, _ in enumerate(sampled_locations)
        ]
    )
    mean_site_env_initial = compute_mean_site(env_initial)
    mean_site_env_altered = compute_mean_site(env_altered)
    # Compute individual environmental variables
    env_sampled_initial = env_initial[np.concatenate(sampled_individuals), :]
    env_sampled_altered = env_altered[np.concatenate(sampled_individuals), :]

    # Compute mean site fitness after environmental change
    mean_site_fitness = np.array(
        [
            np.mean(
                fitness_altered[[ts.node(i).individual for i in sampled_nodes[site]]]
            )
            for site, _ in enumerate(sampled_locations)
        ]
    )
    # Simplify the tree-sequence and get genotypes
    sts = ts.simplify(np.concatenate(sampled_nodes))
    new_nodes = [x for x in sts.samples().reshape(num_individuals * 2, num_sites).T]
    allele_freqs = allele_frequencies(sts, new_nodes).T
    genotype_sampled = genotype_matrix(sts)
    allele_var_mask = allele_freqs.std(axis=0) > 0
    genotype_var_mask = np.var(genotype_sampled, axis=0) > 0
    valid_snps = allele_var_mask | genotype_var_mask  # Logical OR operation

    # Apply the mask
    allele_freqs = allele_freqs[:, valid_snps]
    genotype_sampled = genotype_sampled[:, valid_snps]
    # Sanity check
    assert len(mean_site_fitness) == num_sites
    P = 2 + 5 if noise else 2
    assert mean_site_env_initial.shape == (num_sites, P)
    assert mean_site_env_altered.shape == (num_sites, P)
    assert env_sampled_initial.shape == (int(num_sites * num_individuals), P)
    assert env_sampled_altered.shape == (int(num_sites * num_individuals), P)
    assert allele_freqs.shape[0] == num_sites
    assert genotype_sampled.shape[0] == int(num_sites * num_individuals)
    assert allele_freqs.shape[1] == genotype_sampled.shape[1]
    return {
        "mean_site_fitness": mean_site_fitness,
        "mean_site_env_initial": mean_site_env_initial,
        "mean_site_env_altered": mean_site_env_altered,
        "allele_freqs": allele_freqs,
        "env_sampled_initial": env_sampled_initial,
        "env_sampled_altered": env_sampled_altered,
        "genotype_sampled": genotype_sampled,
        "num_sites": num_sites,
        "num_individuals": num_individuals,
    }


def find_latent_factors(mat):
    pca_model = sm.PCA(mat, standardize=False, demean=True, ncomp=10)
    cumvars = np.cumsum(pca_model.eigenvals / pca_model.eigenvals.sum())
    return np.where(cumvars > 0.70)[0][0]


def main(seed, infile, outfile):
    ts = tskit.load(infile)
    print("Ne", ts.num_individuals)
    metadata = pd.DataFrame(ts.metadata["SLiM"]["user_metadata"])
    print("Mean fitness", metadata.FITNESS.mean())
    print("nQTLs", ts.num_sites)
    causal_dataset = simulate_dataset(ts, seed, False)
    empirical_dataset = simulate_dataset(ts, seed, True)
    assert (
        np.linalg.norm(
            causal_dataset["mean_site_fitness"] - empirical_dataset["mean_site_fitness"]
        )
        < 1e-5
    )
    # Compute genomic offsets (causal dataset)
    data = {
        "Shifted_fitness": causal_dataset["mean_site_fitness"],
        "Seed": seed,
        "Population": np.arange(0, causal_dataset["num_sites"]),
        "Num_QTLs": causal_dataset["genotype_sampled"].shape[1],
    }
    model = go.RONA()
    model.fit(causal_dataset["allele_freqs"], causal_dataset["mean_site_env_initial"])
    data["Causal_RONA"] = model.genomic_offset(
        causal_dataset["mean_site_env_initial"], causal_dataset["mean_site_env_altered"]
    )
    # RDA
    model = go.RDA(n_latent_factors=find_latent_factors(causal_dataset["allele_freqs"]))
    model.fit(causal_dataset["allele_freqs"], causal_dataset["mean_site_env_initial"])
    data["Causal_RDA"] = model.genomic_offset(
        causal_dataset["mean_site_env_initial"], causal_dataset["mean_site_env_altered"]
    )
    # Gradient forest
    model = go.GradientForestGO(n_trees=2000)
    model.fit(causal_dataset["allele_freqs"], causal_dataset["mean_site_env_initial"])
    data["Causal_GradientForest"] = model.genomic_offset(
        causal_dataset["mean_site_env_initial"], causal_dataset["mean_site_env_altered"]
    )
    # GeometricGO
    model = go.GeometricGO(
        n_latent_factors=find_latent_factors(causal_dataset["genotype_sampled"])
    )
    model.fit(causal_dataset["genotype_sampled"], causal_dataset["env_sampled_initial"])
    offset = model.genomic_offset(
        causal_dataset["env_sampled_initial"], causal_dataset["env_sampled_altered"]
    )
    # We have to average across sites for a 'fair' comparison
    data["Causal_GeometricGO"] = offset.reshape(
        causal_dataset["num_sites"], causal_dataset["num_individuals"]
    ).mean(axis=1)
    # Compute genomic offsets (with putatively adaptive alleles)
    Y = empirical_dataset["genotype_sampled"]
    data["Num_SNPs"] = Y.shape[1]
    K = find_latent_factors(Y)
    lfmm = go.RidgeLFMM(n_latent_factors=K, lambda_=1e-5)
    lfmm.fit(Y, empirical_dataset["env_sampled_initial"])
    fscores, pvalues = lfmm.f_test(
        Y, empirical_dataset["env_sampled_initial"], genomic_control=False
    )
    thresholds = [1e-3, 0.1]
    categories = ["strict", "relax"]
    for thres, category in zip(thresholds, categories):
        print("Category", category)
        mask = pvalues * Y.shape[1] < thres
        data[f"Num_candidates_{category}"] = mask.sum()
        print("Putatively adaptive loci", mask.sum())
        model = go.RONA()
        model.fit(
            empirical_dataset["allele_freqs"][:, mask],
            empirical_dataset["mean_site_env_initial"],
        )
        data[f"Empirical_{category}_RONA"] = model.genomic_offset(
            empirical_dataset["mean_site_env_initial"],
            empirical_dataset["mean_site_env_altered"],
        )
        # RDA
        model = go.RDA(
            n_latent_factors=find_latent_factors(
                empirical_dataset["allele_freqs"][:, mask]
            )
        )
        model.fit(
            empirical_dataset["allele_freqs"][:, mask],
            empirical_dataset["mean_site_env_initial"],
        )
        data[f"Empirical_{category}_RDA"] = model.genomic_offset(
            empirical_dataset["mean_site_env_initial"],
            empirical_dataset["mean_site_env_altered"],
        )
        # Gradient forest
        model = go.GradientForestGO(n_trees=2000)
        model.fit(
            empirical_dataset["allele_freqs"][:, mask],
            empirical_dataset["mean_site_env_initial"],
        )
        data[f"Empirical_{category}_GradientForest"] = model.genomic_offset(
            empirical_dataset["mean_site_env_initial"],
            empirical_dataset["mean_site_env_altered"],
        )
        # GeometricGO
        model = go.GeometricGO(
            n_latent_factors=find_latent_factors(
                empirical_dataset["genotype_sampled"][:, mask]
            )
        )
        model.fit(
            empirical_dataset["genotype_sampled"][:, mask],
            empirical_dataset["env_sampled_initial"],
        )
        offset = model.genomic_offset(
            empirical_dataset["env_sampled_initial"],
            empirical_dataset["env_sampled_altered"],
        )
        # We have to average across sites for a 'fair' comparison
        data[f"Empirical_{category}_GeometricGO"] = offset.reshape(
            empirical_dataset["num_sites"], empirical_dataset["num_individuals"]
        ).mean(axis=1)
    # Save results
    data = pd.DataFrame(data)
    data.to_csv(outfile, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise ValueError("Usage: python %s <seed> <infile> <outfile>" % sys.argv[0])
    seed = int(sys.argv[1])
    infile = sys.argv[2]
    outfile = sys.argv[3]
    main(seed, infile, outfile)
