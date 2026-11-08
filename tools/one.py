import sys, time, json
import numpy as np
sys.path.insert(0, '.')
from cfd import solve_cavity
re=float(sys.argv[1]); t=time.time()
f,steps,change,div=solve_cavity(re,n=50,max_steps=400000)
np.save(f'case_{int(re)}.npy',f)
json.dump({'re':re,'steps':int(steps),'steady_change':float(change),'rms_div':float(div),'sec':time.time()-t},open(f'case_{int(re)}.json','w'))
print(f'Re={re:.0f} steps={steps} change={change:.2e} div={div:.1e} {time.time()-t:.0f}s',flush=True)
