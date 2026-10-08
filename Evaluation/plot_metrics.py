"""
Module: plot_metrics.py
Description: Generates statistical visualizations for model evaluation, 
             including paired boxplots (Teacher vs Student) and capacity 
             scaling plots (metrics across parameter counts).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

# Global visual configuration for Matplotlib
plt.rcParams.update({
    'font.size': 16, 
    'axes.titlesize': 20, 
    'axes.labelsize': 18, 
    'xtick.labelsize': 16, 
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'legend.title_fontsize': 18
})

def generate_paired_boxplots(teacher_csv: str, student_csv: str, output_path: str, threshold: float = 0.1) -> None:
    """
    Generates a 4x4 grid of stratified paired boxplots comparing Teacher and Student metrics.
    Calculates Wilcoxon signed-rank tests for statistical significance.
    """
    if os.path.exists(teacher_csv) and os.path.exists(student_csv):
        df_teacher = pd.read_csv(teacher_csv)
        df_student = pd.read_csv(student_csv)
        
        student_name = df_student['Model'].iloc[0] if 'Model' in df_student.columns else "Student"
        
        merge_cols = ["Subject"]
        if "Size" in df_teacher.columns and "Size" in df_student.columns:
            merge_cols.append("Size")
            
        df_merged = pd.merge(df_teacher, df_student, on=merge_cols, suffixes=('_Teacher', '_Student'))
        
        if "Size" not in df_merged.columns:
            if "Size_Teacher" in df_merged.columns:
                df_merged["Size"] = df_merged["Size_Teacher"]
            elif "Size_Student" in df_merged.columns:
                df_merged["Size"] = df_merged["Size_Student"]
            else:
                df_merged["Size"] = "Unknown"

        metrics_to_analyze = [
            {"name": "Global Dice", "col_base": "Global_Dice"},
            {"name": "Lesion Dice", "col_base": "Lesion_Dice"},
            {"name": "Lesion F1",   "col_base": "Lesion_F1"},
            {"name": "Average Surface Distance (mm)", "col_base": "Average_Surface_Distance_mm"}
        ]

        size_categories = [
            {"label": "Total (All Sizes)", "val": None},
            {"label": "Large Lesions (L)", "val": "L"},
            {"label": "Medium Lesions (M)", "val": "M"},
            {"label": "Small Lesions (S)", "val": "S"}
        ]

        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        
        for row_idx, cat in enumerate(size_categories):
            cat_label = cat["label"]
            cat_val = cat["val"]
            
            df_subset = df_merged if cat_val is None else df_merged[df_merged['Size'] == cat_val]

            for col_idx, m in enumerate(metrics_to_analyze):
                ax = axes[row_idx, col_idx]
                t_col = f'{m["col_base"]}_Teacher'
                s_col = f'{m["col_base"]}_Student'
                
                if t_col in df_merged.columns and s_col in df_merged.columns:
                    df_valid = df_subset.dropna(subset=[t_col, s_col]).copy()
                    
                    if not df_valid.empty:
                        diffs = df_valid[s_col] - df_valid[t_col]
                        
                        if len(df_valid) < 3 or np.all(diffs == 0):
                            p_value = 1.0
                            is_sig = False
                        else:
                            _, p_value = wilcoxon(df_valid[t_col], df_valid[s_col])
                            is_sig = p_value < 0.05
                            
                        df_melted = df_valid.melt(
                            id_vars=['Subject'], 
                            value_vars=[t_col, s_col], 
                            var_name='Model', 
                            value_name=f'{m["name"]} Score'
                        )
                        df_melted['Model'] = df_melted['Model'].replace({t_col: 'Teacher', s_col: student_name})
                        
                        sns.boxplot(x='Model', y=f'{m["name"]} Score', hue='Model', data=df_melted, 
                                    palette={"Teacher": "#8da0cb", student_name: "#fc8d62"}, 
                                    width=0.4, boxprops={'alpha': 0.6}, legend=False, ax=ax)
                        sns.stripplot(x='Model', y=f'{m["name"]} Score', data=df_melted, color=".25", alpha=0.3, jitter=True, zorder=0, ax=ax)
                        
                        sig_text = "Significant" if is_sig else "Not Significant"
                        ax.set_title(f"{m['name']} - {cat_label}\np-val: {p_value:.4f} ({sig_text})", fontsize=11)
                        
                        if "Average Surface Distance" in m["name"]:
                            ax.set_ylim(auto=True)
                            current_vals = df_valid[[t_col, s_col]].values.flatten()
                            current_vals = current_vals[~np.isnan(current_vals)]
                            if len(current_vals) > 0:
                                ax.set_ylim(0.05, np.percentile(current_vals, 75) * 1.5)
                        else:
                            ax.set_ylim(-0.05, 1.05)
                            
                        ax.set_ylabel(f'{m["name"]} Score', fontsize=10)
                        ax.set_xlabel('', fontsize=10)
                        ax.grid(axis='y', linestyle='--', alpha=0.5)
                    else:
                        ax.set_title(f"{m['name']} - {cat_label}\n(No Data)", fontsize=12)
                        ax.axis('off')
                else:
                    ax.set_title(f"{m['name']}\n(Data Missing)", fontsize=12)
                    ax.axis('off')

        plt.tight_layout(pad=4.0)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"-> Saved 4x4 paired boxplot grid to '{output_path}'")
        
    else:
        print("Error: Could not find one or both of the provided CSV files.")
        
    return None

def plot_capacity_scaling(csv_files: list, output_path: str, plot_style: str = 'bands') -> None:
    """
    Generates a 2x2 grid showing metric trends across model sizes (parameter counts).
    Supports either confidence 'bands' or error 'bars'.
    """
    param_dict = {
        'Femto': 0.05 * 1e6, 'Pico': 0.21 * 1e6, 'Nano': 0.84 * 1e6,
        'ExtraExtraLight': 2.61 * 1e6, 'ExtraLight': 4.72 * 1e6,
        'Light': 10.44 * 1e6, 'Small': 16.39 * 1e6, 'Medium': 35.27 * 1e6,
        'Large':  52.91 * 1e6, 'Teacher': 102.35 * 1e6
    }

    dfs = []
    for file in csv_files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            raw_model_name = str(df['Model'].iloc[0]).strip()
            
            matched_key = next((k for k in param_dict.keys() if k.lower() == raw_model_name.lower()), None)
            
            if matched_key is not None:
                df['Model'] = matched_key
                df['Params_M'] = param_dict[matched_key]
                
                df_all = df.copy()
                df_all['Size'] = 'All'
                dfs.append(pd.concat([df, df_all], ignore_index=True))
                
    if len(dfs) > 0:
        df_total = pd.concat(dfs, ignore_index=True)
        metrics = ['Global_Dice', 'Lesion_Dice', 'Lesion_F1', 'Average_Surface_Distance_mm']
        df_melted = df_total.melt(id_vars=['Subject', 'Model', 'Params_M', 'Size'], 
                                  value_vars=metrics, var_name='Metric', value_name='Score')

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        titles = ['Global Dice Score', 'Lesion Dice Score', 'Lesion F1 Score', 'Average Surface Distance (mm)']
        
        size_palette = {'All': 'black', 'L': '#1f77b4', 'M': '#ff7f0e', 'S': '#2ca02c'}
        marker_palette = {'All': 'o', 'L': 'X', 'M': 's', 'S': '^'} 
        sizes_to_plot = ['All', 'L', 'M', 'S']

        for i, metric in enumerate(metrics):
            df_sub = df_melted[df_melted['Metric'] == metric]
            df_sub_plot = df_sub[df_sub['Model'] != 'Teacher']
            
            for size in sizes_to_plot:
                df_size = df_sub_plot[df_sub_plot['Size'] == size]
                if not df_size.empty:
                    errorbar_setting = None if size == 'All' else ('ci', 95)
                    
                    if plot_style == 'bands':
                        sns.lineplot(
                            data=df_size, x='Params_M', y='Score', color=size_palette[size],
                            marker=marker_palette[size], linestyle='', err_style='band',                       
                            errorbar=errorbar_setting, err_kws={'alpha': 0.2, 'zorder': 1},    
                            markersize=10, alpha=0.6, label=size if i == 0 else None,         
                            ax=axes[i], zorder=4                                
                        )
                    else: # bars
                        sns.lineplot(
                            data=df_size, x='Params_M', y='Score', color=size_palette[size],
                            marker=marker_palette[size], linestyle='', err_style='bars',
                            errorbar=errorbar_setting, err_kws={'linewidth': 3},
                            markersize=14, alpha=0.6, label=size if i == 0 else None,
                            ax=axes[i]
                        )
            
            for size in sizes_to_plot:
                teacher_mask = (df_sub['Model'] == 'Teacher') & (df_sub['Size'] == size)
                if not df_sub[teacher_mask].empty:
                    teacher_ref = df_sub[teacher_mask]['Score'].mean()
                    axes[i].axhline(y=teacher_ref, color=size_palette[size], linestyle='--', 
                                    linewidth=3, alpha=0.7, zorder=0)
            
            axes[i].set_title(titles[i], fontweight='bold', pad=15)
            axes[i].set_xlabel("Number of Parameters")
            axes[i].set_ylabel("Score" if i < 3 else "Distance (mm)")
            axes[i].set_xscale('log')
            axes[i].grid(True, which="both", ls="--", alpha=0.5)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.98), title="Lesion category")
        
        if axes[0].get_legend() is not None:
            axes[0].get_legend().remove()

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(output_path, format="pdf", bbox_inches='tight', dpi=1200)
        print(f"PDF Plot saved successfully to '{output_path}'.")
        
    return None

if __name__ == "__main__":
    # Example execution
    generate_paired_boxplots(
        teacher_csv="evaluation_results/Evaluation_Teacher.csv",
        student_csv="evaluation_results/Evaluation_Nano.csv",
        output_path="evaluation_results/Teacher_vs_Nano_Boxplots.png"
    )