#!/bin/env python
"""
Populate the SLiM simulations with initial diversity.
"""
import sys

import msprime, pyslim, tskit
import numpy as np


def overlay_mutations(
    rng: np.random.RandomState, ots: tskit.TreeSequence, slim_type: int, num_loci: int
) -> tskit.TreeSequence:
    """Add L mutations of stype slim type as if they had evolved neutrally."""
    next_id = pyslim.next_slim_mutation_id(ots)
    mut_model = msprime.SLiMMutationModel(type=slim_type, next_id=next_id)
    mu = 1e-10
    n_mutations = 0
    while n_mutations < num_loci:
        seed = rng.integers(1, 2 ^ 32 - 1, 1)
        ots = msprime.sim_mutations(
            ots, rate=mu, model=mut_model, keep=True, random_seed=seed
        )
        # Count number of mutations
        n_mutations = np.sum(
            [
                x["mutation_type"] == slim_type
                for site in ots.sites()
                for mut in site.mutations
                for x in mut.metadata["mutation_list"]
            ]
        )
        mu = mu * 2
    num_exceeded = n_mutations - num_loci
    # This code assumes probability of two mutations falling in
    # same site is zero
    candidates = list(
        set(
            site.id
            for site in ots.sites()
            for mut in site.mutations
            for x in mut.metadata["mutation_list"]
            if x["mutation_type"] == slim_type
        )
    )
    discard_ids = rng.choice(candidates, num_exceeded, replace=False)
    return ots.delete_sites(discard_ids)


def main(seed, outfile):
    rng = np.random.default_rng(seed)
    # Define constant parameters of the neutral burn-in simulation
    # I chose this value to match the equilibrium Ne after local adaptation
    # in the SLiM simulation. I want to avoid bottleneck effects.
    Ne0 = 2_000 # Ancestral population size
    sequence_length = 1e8
    recombination_rate = 1e-8
    # Generate initial diversity via msprime
    print("Running coalescence simulation...")
    ots = msprime.sim_ancestry(
        samples=Ne0,
        population_size=Ne0,
        random_seed=seed,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
    )
    # Annotate tree-sequence with SLiM metadata
    ots = pyslim.annotate(ots, model_type="nonWF", tick=1, stage="late")
    print(ots)
    print("Adding standing genetic variation for QTLs of two traits...")
    ots = overlay_mutations(rng, ots, 2, 200)
    ots = overlay_mutations(rng, ots, 3, 200)
    effect_sizes = rng.choice([-0.01, 0.01], size=ots.num_sites)
    # Mutable copy of the tree-sequence object
    tables = ots.tables
    tables.mutations.clear()
    for m in ots.mutations():
        site_id = m.site
        md_list = m.metadata["mutation_list"]
        slim_ids = m.derived_state.split(",")
        assert len(slim_ids) == len(md_list)
        # For simplicity, we'll ignore recurrent mutations
        # in the same QTL (and consider only the first one)
        md_list[0]["selection_coeff"] = effect_sizes[site_id]
        # We'll consider the rest of the mutations, if any, neutral
        for md in md_list[1:]:
            md["mutation_type"] = 1
        # Append the edited mutations
        _ = tables.mutations.append(m.replace(metadata={"mutation_list": md_list}))
    assert tables.mutations.num_rows == ots.num_mutations
    # Finish editing the metadata:
    ts_metadata = tables.metadata
    ts_metadata["SLiM"]["spatial_dimensionality"] = "xy"  # 2D landscape
    tables.metadata = ts_metadata
    ots = tables.tree_sequence()
    print(ots)
    ots.dump(outfile)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python initial_diversity.py <seed> <output>")
    main(int(sys.argv[1]), sys.argv[2])
