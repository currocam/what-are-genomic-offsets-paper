library("tidyverse")

data <- read_csv("genomic_offsets.csv")

causal <- library("tidyverse")

data <- read_csv("genomic_offsets.csv")

causal <- data |>
    pivot_longer(
        cols = starts_with("Causal"),
        names_to = "Method", 
        values_to = "Offset"
    ) |>
    mutate(Method = sub(".*_", "", Method)) |>
    group_by(Seed, Method) |>
    summarise(
        Pearson = cor(-log(Shifted_fitness), Offset),
        Category = "Causal",
        .groups = "drop"
    )

relax <- data |>
    pivot_longer(
        cols = starts_with("Empirical_relax"),
        names_to = "Method", 
        values_to = "Offset"
    ) |>
    mutate(Method = sub(".*_", "", Method)) |>
    group_by(Seed, Method) |>
    summarise(
        Pearson = cor(-log(Shifted_fitness), Offset),
        Category = "Relax",
        .groups = "drop"
    )

strict <- data |>
    pivot_longer(
        cols = starts_with("Empirical_strict"),
        names_to = "Method", 
        values_to = "Offset"
    ) |>
    mutate(Method = sub(".*_", "", Method)) |>
    group_by(Seed, Method) |>
    summarise(
        Pearson = cor(-log(Shifted_fitness), Offset),
        Category = "Strict",
        .groups = "drop"
    )

data2 <- bind_rows(causal, relax, strict)


relax <- data |>
    pivot_longer(
        cols = starts_with("Empirical_relax"),
        names_to = "Method", 
        values_to = "Offset"
    ) |>
    mutate(Method = sub(".*_", "", Method)) |>
    group_by(Seed, Method) |>
    summarise(
        Pearson = cor(-log(Shifted_fitness), Offset),
        Category = "Relax",
        .groups = "drop"
    )

strict <- data |>
    pivot_longer(
        cols = starts_with("Empirical_strict"),
        names_to = "Method", 
        values_to = "Offset"
    ) |>
    mutate(Method = sub(".*_", "", Method)) |>
    group_by(Seed, Method) |>
    summarise(
        Pearson = cor(-log(Shifted_fitness), Offset),
        Category = "Strict",
        .groups = "drop"
    )

data2 <- bind_rows(causal, relax, strict)

p1 <- data2 |>
    ggplot(aes(x = Method, y = Pearson, color = Category)) +
    geom_boxplot() +  # Adjust width and remove outliers
    scale_y_continuous(limits = c(0.0, 1.0), expand = c(0, 0)) +  # Ensure axis starts at 0
    scale_color_brewer(palette = "Dark2") +  # Use a colorblind-friendly palette
    theme_minimal(base_size = 14) +  # Use a cleaner theme
    xlab("") +
    ylab("Pearson correlation")

ggsave("plot.pdf", p1)

