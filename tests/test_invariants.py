import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from scipy.integrate import quad
from scipy.stats import binom
from experiment import endpoint_score,score_at,safe_speed,stop_distance,quantile,make_bounds,evaluate,population,VMAX,one_run

class ScientificInvariants(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(909)
        self.y=self.rng.uniform(.02,1,(200,2));self.p=self.rng.uniform(.02,1,(200,2));self.s=self.rng.uniform(.01,.4,(200,2))
    def test_endpoint_supremum(self):
        grid=np.linspace(0,VMAX,10001)
        scores=np.column_stack([score_at(self.y,self.p,self.s,v) for v in grid])
        np.testing.assert_allclose(endpoint_score(self.y,self.p,self.s),np.maximum(0,scores.max(axis=1)),atol=1e-12)
    def test_speed_feasibility(self):
        d=self.rng.uniform(.001,8,200);v=safe_speed(self.y,d)
        self.assertTrue(np.all(stop_distance(self.y,v)<=d+1e-12))
        mask=v<VMAX-1e-8
        np.testing.assert_allclose(stop_distance(self.y,v)[mask],d[mask],atol=1e-12)
    def test_containment_implies_stop(self):
        q=endpoint_score(self.y,self.p,self.s);coef=self.p+self.s*q[:,None]
        for _ in range(10):
            d=self.rng.uniform(.001,8,200);v=safe_speed(coef,d)
            self.assertTrue(np.all(stop_distance(self.y,v)<=d+1e-10))
    def test_rank_and_infinity(self):
        self.assertEqual(quantile(np.arange(19),.05),18)
        self.assertTrue(np.isinf(quantile(np.arange(18),.05)))
        self.assertEqual(quantile(np.arange(511),.05),486)
        self.assertEqual(quantile(np.arange(511),.05,True),int(binom.ppf(.95,511,.95)))
    def test_numerical_dynamics(self):
        for tau,beta in self.y[:30]:
            a=1/(2*beta);v=float(self.rng.uniform(.01,3));tstop=tau+v/a
            area=quad(lambda t:v if t<tau else max(0,v-a*(t-tau)),0,tstop,points=[tau],epsabs=1e-12)[0]
            self.assertAlmostEqual(area,tau*v+beta*v*v,places=10)
    def test_policy_threshold(self):
        d=self.rng.uniform(.01,3,200);vs=safe_speed(self.y,d)
        r=np.where(stop_distance(self.y,np.full(200,VMAX))<=d,0,score_at(self.y,self.p,self.s,vs))
        coef=self.p+self.s*np.maximum(r,0)[:,None]
        self.assertTrue(np.all(stop_distance(self.y,safe_speed(coef,d))<=d+1e-10))
    def test_grid_interpolation(self):
        # Exact positive coefficients dominate true curves; convex interpolation remains safe.
        b=('grid',np.full(21,100.));d=self.rng.uniform(.01,3,200)
        metrics,_,_=evaluate(self.y,self.p,self.s,d,b)
        self.assertEqual(metrics['curve_failure'],0.)
        self.assertEqual(metrics['stop_failure'],0.)
    def test_stream_repeatability(self):
        for s in ['gaussian','heteroscedastic','heavy_tail','mixture']:
            a=population(100,np.random.default_rng(17),s);b=population(100,np.random.default_rng(17),s)
            for aa,bb in zip(a,b):np.testing.assert_array_equal(aa,bb)
if __name__=='__main__':unittest.main()
