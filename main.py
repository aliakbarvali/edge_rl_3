# تست سریع: فقط چند صد timestep و ۲ epoch BC، فقط برای گرفتن خطاهای API/نسخه
import sys
sys.path.insert(0, '.')
from algorithms.ppo.train import main

main(total_timesteps=2000, bc_epochs=2, window_hours=1.0, bc_max_ticks=50)