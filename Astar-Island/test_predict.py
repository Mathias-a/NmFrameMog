import sys
from pathlib import Path
sys.path.insert(0, str(Path('benchmark/src').resolve()))
from astar_twin.solver.predict.posterior_mc import predict_all_seeds

print(predict_all_seeds.__annotations__)
