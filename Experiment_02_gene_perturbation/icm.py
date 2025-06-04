import pandas as pd
import json
from collections import defaultdict

# import causalicp as icp

int_data = pd.read_csv("data/interventional_data.csv", index_col=0)
obs_data = pd.read_csv("data/observational_data.csv", index_col=0)
int_pos_data = pd.read_csv("data/interventional_position_data.csv", index_col=0)
intervened_genes = set(int_pos_data["Mutant"])
gene_names = list(obs_data.columns)

genes = [
    "YMR104C",
    "YMR103C",
    "YPL273W",
    "YMR321C",
    "YCL040W",
    "YCL042W",
    "YLL019C",
    "YLL020C",
    "YMR186W",
    "YPL240C",
    "YDR074W",
    "YBR126C",
    "YMR173W",
    "YMR173W-A",
    "YGR162W",
    "YGR264C",
    "YOR027W",
    "YJL077C",
    "YJL115W",
    "YLR170C",
    "YOR153W",
    "YDR011W",
    "YLR270W",
    "YLR345W",
    "YOR153W",
    "YBL005W",
    "YJL141C",
    "YNR007C",
    "YAL059W",
    "YPL211W",
    "YLR263W",
    "YKL098W",
    "YGR271C-A",
    "YDR339C",
    "YLL019C",
    "YGR130C",
    "YCL040W",
    "YML100W",
    "YMR310C",
    "YOR224C",
]

print(len(set(intervened_genes).intersection(set(genes))))


def find_SIE_candidates(target_gene, int_data, int_pos_data, all_genes):

    SIE_candidates = []

    for candidate_gene in all_genes:
        if candidate_gene == target_gene:
            continue

        # Lấy chỉ số các sample can thiệp vào candidate_gene
        int_idx = int_pos_data[int_pos_data["Mutant"] == candidate_gene].index
        if len(int_idx) == 0:
            continue

        for idx in int_idx:
            if idx not in int_data.index:
                continue

            x_val = int_data.loc[idx, candidate_gene]
            y_val = int_data.loc[idx, target_gene]

            # Dữ liệu không can thiệp vào candidate_gene
            mask_other = int_pos_data["Mutant"] != candidate_gene
            other_idx = int_pos_data[mask_other].index.intersection(int_data.index)

            x_range = int_data.loc[other_idx, candidate_gene]
            y_range = int_data.loc[other_idx, target_gene]

            # Điều kiện SIE
            x_extreme = (x_val < x_range.min()) or (x_val > x_range.max())
            y_extreme = (y_val < y_range.min()) or (y_val > y_range.max())

            if x_extreme and y_extreme:
                SIE_candidates.append(candidate_gene)
                # break  # Chỉ cần một sample là đủ

    return SIE_candidates


all_genes = list(int_data.columns)
sie_results = {}

for i, gene in enumerate(all_genes):
    print(f"[{i+1}/{len(all_genes)}] Checking SIE for gene: {gene}")

    causal_candidates = find_SIE_candidates(gene, int_data, int_pos_data, all_genes)
    print(f"Possible causal genes for {gene} (via SIE): {causal_candidates}")
    sie_results[gene] = causal_candidates

# === Save to JSON ===
with open("sie_results.json", "w") as f:
    json.dump(sie_results, f, indent=2)


def find_icp_causal_candidates_subset(
    obs_data,
    int_data,
    int_pos_data,
    target_gene,
    all_genes,
    sie_results,
    alpha=0.05,
    verbose=False,
    n_repeats=20,
    sample_frac=0.8,
    min_accept_count=1,
):
    selection_counts = defaultdict(int)

    if target_gene not in obs_data.columns:
        raise ValueError(f"Target gene {target_gene} not in dataset.")

    # Get list of SIE-valid genes for this target
    sie_valid_genes = sie_results.get(target_gene, [])

    for gene in sie_valid_genes:
        # if gene == target_gene:
        #     continue

        # if gene not in sie_valid_genes:
        #     continue  # Skip if SIE not passed

        accept_count = 0

        for repeat in range(n_repeats):
            obs_sample = obs_data.sample(
                frac=sample_frac, replace=False, random_state=repeat
            )
            int_sample = int_data.sample(
                frac=sample_frac, replace=False, random_state=repeat
            )

            # Drop rows where gene is mutated
            intervened_idx = int_pos_data[int_pos_data["Mutant"] == gene].index
            int_sample_filtered = int_sample.drop(index=intervened_idx, errors="ignore")

            # print(int_sample_filtered)

            obs_copy = obs_sample.copy()
            obs_copy["Environment"] = 0

            int_copy = int_sample_filtered.copy()
            int_copy["Environment"] = 1

            merged_data = pd.concat([obs_copy, int_copy], axis=0).reset_index(drop=True)

            temp_df = merged_data[[gene, target_gene, "Environment"]].copy()
            target_index = temp_df.columns.get_loc(target_gene)

            data_list = []
            for env in temp_df["Environment"].unique():
                env_data = (
                    temp_df[temp_df["Environment"] == env]
                    .drop(columns=["Environment"])
                    .values
                )
                data_list.append(env_data)

            try:
                result = icp.fit(
                    data_list, target=target_index, alpha=alpha, verbose=verbose
                )
            except Exception as e:
                if verbose:
                    print(f"ICP failed on repeat {repeat} for gene {gene}: {e}")
                continue

            if len(result.accepted_sets) > 0:
                accept_count += 1

        if accept_count >= min_accept_count:
            selection_counts[gene] = accept_count

    # Sort by descending number of selections
    sorted_genes = sorted(selection_counts.items(), key=lambda x: x[1], reverse=True)

    return sorted_genes


causal_genes_ICP_dict = {}

candidate_causes = list(set(intervened_genes))

for i, target_gene in enumerate(gene_names[1000:2000]):
    print(f"Target gene {i} : {target_gene}")

    try:
        # Lọc các gene khác để xét làm nguyên nhân
        all_genes = [g for g in candidate_causes if g != target_gene]

        # Tìm các causal gene được chọn nhiều nhất qua các subset
        causal_ranked = find_icp_causal_candidates_subset(
            obs_data=obs_data,
            int_data=int_data,
            int_pos_data=int_pos_data,
            target_gene=target_gene,
            all_genes=all_genes,
            sie_results=sie_results,
            alpha=0.01,
            verbose=False,
            n_repeats=30,
            sample_frac=0.7,
            min_accept_count=1,
        )

        # Lưu top 2 causal genes (hoặc ít hơn nếu không đủ)
        top_causal_genes = [gene for gene, count in causal_ranked[:2]]

        if len(top_causal_genes) > 0:
            causal_genes_ICP_dict[target_gene] = top_causal_genes
            print(f"Top causal for {target_gene}: {top_causal_genes}")

    except Exception as e:
        print(f"Error at index {i} ({target_gene}): {e}")

with open("ICP_causal_intervened_genes_resample_1000_2000.json", "w") as f:
    json.dump(causal_genes_ICP_dict, f, indent=2)
