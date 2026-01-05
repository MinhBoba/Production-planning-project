"""
Visualization and reporting for Tabu Search solutions.
Generates detailed production reports and visual schedules.
Designed for both console output and web API integration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Any
import json
import base64
from io import BytesIO


class SolutionVisualizer:
    """Generates reports and visualizations for solutions."""

    def __init__(self, input_data, precomputed: Dict):
        """
        Parameters
        ----------
        input_data : InputData
            Problem data
        precomputed : Dict
            Precomputed data from evaluator
        """
        self.input = input_data
        self.precomputed = precomputed
        
        # Color schemes for consistent visualization
        self.color_schemes = {
            'status': {
                'On-Time': '#28a745',        # Green
                'Fulfilled Late': '#ffc107',  # Yellow
                'Partially Fulfilled': '#fd7e14',  # Orange
                'Not Fulfilled': '#dc3545'    # Red
            },
            'background': {
                'On-Time': '#d4edda',
                'Fulfilled Late': '#fff3cd',
                'Partially Fulfilled': '#f8d7da',
                'Not Fulfilled': '#f8d7da'
            }
        }

    # ================================================================
    # CONSOLE OUTPUT METHODS
    # ================================================================

    def print_solution_summary(self, solution: Dict):
        """Print high-level summary of solution costs and metrics."""
        if not solution:
            print("No solution available to summarize.")
            return

        print("\n" + "=" * 70)
        print(" " * 25 + "SOLUTION SUMMARY")
        print("=" * 70)

        setup_cost = len(solution.get("changes", {})) * self.input.param["Csetup"]
        final_backlog_qty = sum(solution.get("final_backlog", {}).values())
        final_late_cost = sum(
            final_backlog_qty * self.input.param["Plate"][s]
            for s in self.input.set["setS"]
        )

        print(f"\n{'Metric':<35} {'Value':>20}")
        print("-" * 70)
        print(f"{'Total Cost':<35} {solution['total_cost']:>20,.2f}")
        print(f"{'  - Setup Cost':<35} {setup_cost:>20,.2f}")
        print(f"{'  - Late Penalty Cost':<35} {final_late_cost:>20,.2f}")
        print(f"{'  - Experience Reward':<35} {solution.get('total_exp', 0):>20,.2f}")
        print("-" * 70)
        print(f"{'Total Backlog Quantity':<35} {final_backlog_qty:>20,.0f} units")
        print(f"{'Number of Setups':<35} {len(solution.get('changes', {})):>20}")
        
        avg_exp = (
            np.mean(list(solution.get("experience", {}).values()))
            if solution.get("experience")
            else 0
        )
        print(f"{'Average Experience':<35} {avg_exp:>20.2f} days")
        print("=" * 70 + "\n")

    def plot_cost_progress(
        self, 
        costs: List[float], 
        save_path: Optional[str] = None,
        return_base64: bool = False
    ) -> Optional[str]:
        """
        Plot cost evolution over iterations.
        
        Parameters
        ----------
        costs : List[float]
            Cost values at each iteration
        save_path : str, optional
            Path to save the figure
        return_base64 : bool
            If True, return base64-encoded image for web API
            
        Returns
        -------
        str or None
            Base64-encoded image if return_base64=True
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        iterations = range(len(costs))
        ax.plot(iterations, costs, linewidth=2, color='#2E86AB', alpha=0.8)
        
        # Add best cost line
        best_cost = min(costs)
        ax.axhline(y=best_cost, color='#06A77D', linestyle='--', 
                   linewidth=2, alpha=0.7, label=f'Best: {best_cost:,.2f}')
        
        # Styling
        ax.set_title('Optimization Progress - Tabu Search', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Total Cost', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='upper right', fontsize=11)
        
        # Format y-axis with thousand separators
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if return_base64:
            return self._fig_to_base64(fig)
        
        plt.show()
        plt.close()
        return None

    def plot_backlog(
        self,
        solution: Dict,
        save_path: Optional[str] = None,
        return_base64: bool = False,
    ) -> Optional[str]:
        """
        Plot final backlog quantity per style as a bar chart.

        Parameters
        ----------
        solution : Dict
            Solution dictionary containing 'final_backlog'
        save_path : str, optional
            Path to save the figure
        return_base64 : bool
            If True, return base64-encoded image

        Returns
        -------
        str or None
            Base64-encoded image if `return_base64=True`
        """
        if not solution:
            print("No solution available to visualize backlog.")
            return None

        backlog = solution.get('final_backlog', {})
        if not backlog:
            print("No backlog data found in solution.")
            return None

        styles = sorted(list(backlog.keys()))
        qtys = [backlog[s] for s in styles]

        fig, ax = plt.subplots(figsize=(max(8, len(styles) * 0.4), 6))
        bars = ax.bar(styles, qtys, color='#2E86AB', edgecolor='black', alpha=0.9)

        ax.set_title('Final Backlog per Style', fontsize=14, fontweight='bold')
        ax.set_ylabel('Quantity', fontsize=12)
        ax.set_xticks(range(len(styles)))
        ax.set_xticklabels(styles, rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        # Annotate bars
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + max(1, h * 0.01), f'{h:,.0f}',
                    ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if return_base64:
            return self._fig_to_base64(fig)

        plt.show()
        plt.close()
        return None

    def visualize_schedule(
        self, 
        solution: Dict,
        save_path: Optional[str] = None,
        return_base64: bool = False
    ) -> Optional[str]:
        """
        Visualize production schedule as Gantt chart.
        
        Parameters
        ----------
        solution : Dict
            Solution with assignment data
        save_path : str, optional
            Path to save the figure
        return_base64 : bool
            If True, return base64-encoded image for web API
            
        Returns
        -------
        str or None
            Base64-encoded image if return_base64=True
        """
        if not solution:
            print("No solution available to visualize.")
            return None

        lines = sorted(list(self.input.set["setL"]))
        styles = sorted(list(self.input.set["setS"]))
        time_periods = sorted(list(self.input.set["setT"]))
        
        # Create figure
        fig, ax = plt.subplots(
            figsize=(max(16, len(time_periods) * 0.5), max(8, len(lines) * 0.6))
        )

        # Generate color palette
        cmap_name = "tab20" if len(styles) <= 20 else "viridis"
        colors = plt.cm.get_cmap(cmap_name, len(styles))
        style_colors = {style: colors(i) for i, style in enumerate(styles)}

        # Plot schedule
        for li, line in enumerate(lines):
            for ti, t in enumerate(time_periods):
                style = solution["assignment"].get((line, t))
                if style:
                    ax.barh(
                        li, 
                        1, 
                        left=t - 0.5,
                        height=0.8,
                        color=style_colors.get(style, 'grey'),
                        edgecolor='white',
                        linewidth=1.5,
                        alpha=0.9
                    )

        # Create legend
        legend_elements = [
            mpatches.Patch(color=style_colors[s], label=s) 
            for s in styles
        ]
        ax.legend(
            handles=legend_elements,
            title="Styles",
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            fontsize=10,
            title_fontsize=12
        )
        
        # Styling
        ax.set_title('Production Schedule - Line Assignment', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Time Period (Day)', fontsize=12)
        ax.set_ylabel('Production Line', fontsize=12)
        ax.set_yticks(range(len(lines)))
        ax.set_yticklabels([f'Line {l}' for l in lines])
        ax.set_xticks(time_periods)
        ax.set_xticklabels(time_periods, rotation=45 if len(time_periods) > 20 else 0)
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        
        plt.tight_layout(rect=[0, 0, 0.92, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if return_base64:
            return self._fig_to_base64(fig)
        
        plt.show()
        plt.close()
        return None

    # ================================================================
    # DETAILED REPORTS - CONSOLE & DATAFRAME OUTPUT
    # ================================================================

    def generate_production_report(self, solution: Dict) -> pd.DataFrame:
        """
        Generate detailed production report.
        
        Returns
        -------
        pd.DataFrame
            Production quantities by line, style, and day
        """
        records = [
            {
                "Day": t,
                "Line": l,
                "Style": solution["assignment"].get((l, t)),
                "Quantity": solution["production"].get(
                    (l, solution["assignment"].get((l, t)), t), 0
                ),
                "SAM_Workload": solution["production"].get(
                    (l, solution["assignment"].get((l, t)), t), 0
                ) * self.precomputed["style_sam"].get(
                    solution["assignment"].get((l, t)), 0
                ),
                "Experience": solution["experience"].get((l, t), 0),
                "Efficiency": solution["efficiency"].get((l, t), 0) * 100,
            }
            for l in self.input.set["setL"]
            for t in self.input.set["setT"]
            if solution["assignment"].get((l, t))
        ]

        return pd.DataFrame(records)

    def generate_order_fulfillment_report(
        self, 
        solution: Dict, 
        order_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate order fulfillment status report.
        
        Parameters
        ----------
        solution : Dict
            Optimized solution
        order_df : pd.DataFrame
            Order data with columns: Style, Demand, DeadlineDay
            
        Returns
        -------
        pd.DataFrame
            Fulfillment report with status for each order
        """
        report_df = order_df.copy()
        
        # Rename columns if needed
        column_mapping = {
            'Style2': 'Style',
            'Sum': 'Demand',
            'Exf-SX': 'DeadlineDay'
        }
        report_df.rename(columns=column_mapping, inplace=True, errors='ignore')
        
        # Initialize tracking columns
        report_df['QtyShipped'] = 0.0
        report_df['CompletionDay'] = np.nan

        sorted_orders = report_df.sort_values('DeadlineDay').index

        # Allocate shipments to orders (FIFO by deadline)
        for t in sorted(self.input.set["setT"]):
            for s in self.input.set["setS"]:
                ship_qty = solution["shipment"].get((s, t), 0)
                if ship_qty <= 1e-6:
                    continue

                unfulfilled_orders = [
                    idx for idx in sorted_orders
                    if report_df.loc[idx, 'Style'] == s
                    and report_df.loc[idx, 'QtyShipped'] < report_df.loc[idx, 'Demand']
                ]

                for idx in unfulfilled_orders:
                    if ship_qty <= 1e-6:
                        break

                    needed = report_df.loc[idx, 'Demand'] - report_df.loc[idx, 'QtyShipped']
                    allocated = min(ship_qty, needed)

                    report_df.loc[idx, 'QtyShipped'] += allocated
                    ship_qty -= allocated

                    # Mark completion
                    if (report_df.loc[idx, 'QtyShipped'] >= report_df.loc[idx, 'Demand'] - 1e-6
                        and pd.isna(report_df.loc[idx, 'CompletionDay'])):
                        report_df.loc[idx, 'CompletionDay'] = t

        # Calculate on-time vs late
        report_df['ShippedOnTime'] = 0.0
        report_df['ShippedLate'] = 0.0

        for idx, order in report_df.iterrows():
            if not pd.isna(order['CompletionDay']):
                if order['CompletionDay'] <= order['DeadlineDay']:
                    report_df.loc[idx, 'ShippedOnTime'] = order['QtyShipped']
                else:
                    report_df.loc[idx, 'ShippedLate'] = order['QtyShipped']

        report_df['Unfulfilled'] = report_df['Demand'] - report_df['QtyShipped']
        
        # Determine status
        def get_status(row):
            if row['Unfulfilled'] > 1:
                return "Partially Fulfilled"
            if not pd.isna(row['CompletionDay']) and row['CompletionDay'] > row['DeadlineDay']:
                return "Fulfilled Late"
            if pd.isna(row['CompletionDay']):
                return "Not Fulfilled"
            return "On-Time"

        report_df['Status'] = report_df.apply(get_status, axis=1)
        report_df['Fulfillment_Pct'] = np.where(
            report_df['Demand'] > 0,
            (report_df['QtyShipped'] / report_df['Demand'] * 100),
            100.0,
        )
        
        # Calculate days late
        report_df['CompletionDay_num'] = pd.to_numeric(
            report_df['CompletionDay'], errors='coerce'
        )
        report_df['Days_Late'] = report_df['CompletionDay_num'] - report_df['DeadlineDay']
        report_df['Days_Late'] = report_df['Days_Late'].apply(
            lambda x: max(0, x) if pd.notna(x) else 0
        )

        return report_df

    def print_detailed_reports(
        self, 
        solution: Dict, 
        order_df: Optional[pd.DataFrame] = None
    ):
        """Generate all detailed production and fulfillment reports."""
        if not solution:
            print("No solution available for reporting.")
            return

        print("\n" + "=" * 70)
        print(" " * 18 + "DETAILED PRODUCTION REPORTS")
        print("=" * 70)

        # Production Report
        prod_report = self.generate_production_report(solution)
        
        if prod_report.empty:
            print("\nNo production data available.")
            return

        all_days = sorted(self.input.set["setT"])

        # Report 1: Production Quantities
        print("\n[1] PRODUCTION QUANTITIES (Units)")
        print("-" * 70)
        prod_pivot = (
            prod_report.pivot_table(
                index=["Line", "Style"],
                columns="Day",
                values="Quantity",
                aggfunc="sum",
                fill_value=0,
            ).reindex(columns=all_days, fill_value=0)
        )
        prod_pivot["Total"] = prod_pivot.sum(axis=1)
        print(prod_pivot.to_string(float_format=lambda x: f'{x:,.0f}'))

        # Report 2: SAM Workload
        print("\n[2] SAM WORKLOAD (Minutes)")
        print("-" * 70)
        sam_pivot = (
            prod_report.pivot_table(
                index=["Line", "Style"],
                columns="Day",
                values="SAM_Workload",
                aggfunc="sum",
                fill_value=0,
            ).reindex(columns=all_days, fill_value=0)
        )
        sam_pivot["Total"] = sam_pivot.sum(axis=1)
        print(sam_pivot.to_string(float_format=lambda x: f'{x:,.0f}'))

        # Report 3: Experience Accumulation
        print("\n[3] EXPERIENCE ACCUMULATION (Days)")
        print("-" * 70)
        exp_pivot = (
            prod_report.pivot_table(
                index=["Line", "Style"],
                columns="Day",
                values="Experience",
                aggfunc="mean",
                fill_value=0,
            ).reindex(columns=all_days, fill_value=0)
        )
        print(exp_pivot.to_string(float_format=lambda x: f'{x:.1f}'))

        # Report 4: Worker Efficiency
        print("\n[4] WORKER EFFICIENCY (%)")
        print("-" * 70)
        eff_pivot = (
            prod_report.pivot_table(
                index=["Line", "Style"],
                columns="Day",
                values="Efficiency",
                aggfunc="mean",
                fill_value=0,
            ).reindex(columns=all_days, fill_value=0)
        )
        print(eff_pivot.to_string(float_format=lambda x: f'{x:.1f}'))

        # Order Fulfillment Reports
        if order_df is not None:
            fulfillment_report = self.generate_order_fulfillment_report(
                solution, order_df
            )
            self._print_fulfillment_summary(fulfillment_report)

    def _print_fulfillment_summary(self, report_df: pd.DataFrame):
        """Print order fulfillment summary."""
        print("\n[5] ORDER FULFILLMENT SUMMARY")
        print("-" * 70)
        
        # Status distribution
        status_counts = report_df['Status'].value_counts()
        total_orders = len(report_df)
        
        print(f"\n{'Status':<25} {'Count':>10} {'Percentage':>15}")
        print("-" * 52)
        for status in ['On-Time', 'Fulfilled Late', 'Partially Fulfilled', 'Not Fulfilled']:
            count = status_counts.get(status, 0)
            pct = (count / total_orders * 100) if total_orders > 0 else 0
            print(f"{status:<25} {count:>10} {pct:>14.1f}%")
        
        # Key metrics
        print("\n" + "-" * 70)
        total_demand = report_df['Demand'].sum()
        total_shipped = report_df['QtyShipped'].sum()
        avg_fulfillment = report_df['Fulfillment_Pct'].mean()
        avg_days_late = report_df[report_df['Days_Late'] > 0]['Days_Late'].mean()
        
        print(f"{'Total Demand':<40} {total_demand:>20,.0f} units")
        print(f"{'Total Shipped':<40} {total_shipped:>20,.0f} units")
        print(f"{'Average Fulfillment Rate':<40} {avg_fulfillment:>20.1f}%")
        if not np.isnan(avg_days_late):
            print(f"{'Average Days Late (late orders only)':<40} {avg_days_late:>20.1f} days")

    # ================================================================
    # WEB API / JSON EXPORT METHODS
    # ================================================================

    def export_solution_json(self, solution: Dict) -> str:
        """
        Export solution summary as JSON for web API.
        
        Returns
        -------
        str
            JSON string with solution metrics
        """
        setup_cost = len(solution.get("changes", {})) * self.input.param["Csetup"]
        final_backlog_qty = sum(solution.get("final_backlog", {}).values())
        final_late_cost = sum(
            final_backlog_qty * self.input.param["Plate"][s]
            for s in self.input.set["setS"]
        )
        avg_exp = (
            np.mean(list(solution.get("experience", {}).values()))
            if solution.get("experience")
            else 0
        )

        data = {
            "summary": {
                "total_cost": float(solution["total_cost"]),
                "setup_cost": float(setup_cost),
                "late_penalty_cost": float(final_late_cost),
                "experience_reward": float(solution.get("total_exp", 0)),
                "total_backlog_qty": float(final_backlog_qty),
                "num_setups": len(solution.get("changes", {})),
                "average_experience": float(avg_exp)
            },
            "costs": {
                "setup": float(setup_cost),
                "late": float(final_late_cost),
                "experience_reward": float(solution.get("total_exp", 0))
            }
        }

        return json.dumps(data, indent=2)

    def export_production_report_json(self, solution: Dict) -> str:
        """
        Export production report as JSON for web API.
        
        Returns
        -------
        str
            JSON string with production data
        """
        report_df = self.generate_production_report(solution)
        
        # Convert to records
        data = {
            "production": report_df.to_dict(orient='records'),
            "summary_by_line": report_df.groupby('Line').agg({
                'Quantity': 'sum',
                'SAM_Workload': 'sum',
                'Experience': 'mean',
                'Efficiency': 'mean'
            }).to_dict(orient='index')
        }

        return json.dumps(data, indent=2, default=str)

    def export_fulfillment_report_json(
        self, 
        solution: Dict, 
        order_df: pd.DataFrame
    ) -> str:
        """
        Export order fulfillment report as JSON for web API.
        
        Returns
        -------
        str
            JSON string with fulfillment data
        """
        report_df = self.generate_order_fulfillment_report(solution, order_df)
        
        # Status distribution
        status_counts = report_df['Status'].value_counts().to_dict()
        total_orders = len(report_df)
        
        data = {
            "fulfillment": report_df.to_dict(orient='records'),
            "summary": {
                "total_orders": total_orders,
                "status_distribution": {
                    status: {
                        "count": int(count),
                        "percentage": float(count / total_orders * 100)
                    }
                    for status, count in status_counts.items()
                },
                "metrics": {
                    "total_demand": float(report_df['Demand'].sum()),
                    "total_shipped": float(report_df['QtyShipped'].sum()),
                    "avg_fulfillment_rate": float(report_df['Fulfillment_Pct'].mean()),
                    "avg_days_late": float(
                        report_df[report_df['Days_Late'] > 0]['Days_Late'].mean()
                    ) if (report_df['Days_Late'] > 0).any() else 0.0
                }
            }
        }

        return json.dumps(data, indent=2, default=str)

    def export_schedule_data_json(self, solution: Dict) -> str:
        """
        Export schedule data as JSON for web visualization.
        
        Returns
        -------
        str
            JSON string with schedule assignment data
        """
        schedule_data = []
        
        for line in self.input.set["setL"]:
            line_schedule = []
            for t in sorted(self.input.set["setT"]):
                style = solution["assignment"].get((line, t))
                if style:
                    line_schedule.append({
                        "day": int(t),
                        "style": style,
                        "quantity": float(solution["production"].get((line, style, t), 0)),
                        "experience": float(solution["experience"].get((line, t), 0)),
                        "efficiency": float(solution["efficiency"].get((line, t), 0))
                    })
            
            schedule_data.append({
                "line": line,
                "schedule": line_schedule
            })

        return json.dumps({"schedules": schedule_data}, indent=2)

    # ================================================================
    # UTILITY METHODS
    # ================================================================

    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string for web API."""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        return image_base64

    def create_visualization_package(
        self, 
        solution: Dict, 
        costs: List[float],
        order_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Create complete visualization package for web API.
        
        Parameters
        ----------
        solution : Dict
            Optimized solution
        costs : List[float]
            Cost progression
        order_df : pd.DataFrame, optional
            Order data for fulfillment report
            
        Returns
        -------
        Dict
            Complete package with all reports and visualizations as JSON/base64
        """
        package = {
            "summary": json.loads(self.export_solution_json(solution)),
            "production_report": json.loads(
                self.export_production_report_json(solution)
            ),
            "schedule_data": json.loads(
                self.export_schedule_data_json(solution)
            ),
            "visualizations": {
                "cost_progress": self.plot_cost_progress(
                    costs, return_base64=True
                ),
                "schedule_gantt": self.visualize_schedule(
                    solution, return_base64=True
                )
            }
        }
        
        if order_df is not None:
            package["fulfillment_report"] = json.loads(
                self.export_fulfillment_report_json(solution, order_df)
            )
        
        return package
