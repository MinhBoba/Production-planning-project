import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import load_input
from metaheuristic.visualizer import SolutionVisualizer
import pickle, os

EXCEL='StandardInput.xlsx'
RES_DIR='result'
os.makedirs(RES_DIR, exist_ok=True)

# Load input and solution
data = load_input(EXCEL)
with open(os.path.join(RES_DIR,'result.pkl'),'rb') as f:
    sol = pickle.load(f)

precomputed = {'style_sam': data.param.get('paramSAM', {})}
vis = SolutionVisualizer(data, precomputed)

# Save schedule image
vis.visualize_schedule(sol, save_path=os.path.join(RES_DIR,'schedule.png'))
print('Saved schedule to', os.path.join(RES_DIR,'schedule.png'))

# Save production report CSV
prod_df = vis.generate_production_report(sol)
prod_df.to_csv(os.path.join(RES_DIR,'production_report.csv'), index=False)
print('Saved production report to', os.path.join(RES_DIR,'production_report.csv'))

# Save backlog chart using visualizer
vis.plot_backlog(sol, save_path=os.path.join(RES_DIR,'backlog.png'))
print('Saved backlog chart to', os.path.join(RES_DIR,'backlog.png'))
