"""Scientific checks for bounded-error calibration and strongest comparator."""
import sys,itertools,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from measurement import np,reconstruct,error_allowance,robust_score,measured_labels
from experiment import endpoint_score,joint_grid_score,quantile

class MeasurementInvariants(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(20261119)
        self.y=self.rng.uniform(.02,1,size=(80,2))
        self.p=self.rng.uniform(.02,1,size=(80,2))
        self.s=self.rng.uniform(.01,.2,size=(80,2))
    def test_identification_inverse(self):
        for u1 in (.3,.75,1.5):
            yhat,_=measured_labels(self.y,self.rng.uniform(size=self.y.shape),0,'symmetric',u1)
            np.testing.assert_allclose(yhat,self.y,rtol=1e-12,atol=1e-12)
    def test_robust_supremum_against_all_error_corners_and_dense_speeds(self):
        v=np.linspace(0,3,10001)
        for u1 in (.3,.75,1.5):
            eta=.025;u2=2.5;u=np.array([u1,u2])
            measurements=self.y[:,0,None]*u+self.y[:,1,None]*u*u
            maximum=np.zeros(len(self.y))
            for signs in itertools.product((-1,1),repeat=2):
                candidate=reconstruct(measurements+eta*np.array(signs),u1,u2)
                e=candidate-self.p
                vals=(e[:,0,None]+v*e[:,1,None])/(self.s[:,0,None]+v*self.s[:,1,None])
                maximum=np.maximum(maximum,vals.max(axis=1))
            np.testing.assert_allclose(robust_score(self.y,self.p,self.s,eta,u1),maximum,rtol=1e-12,atol=1e-12)
    def test_true_score_dominated_for_all_bounded_errors(self):
        for u1 in (.3,.75,1.5):
            for profile in ('symmetric','under_read'):
                estimate,_=measured_labels(self.y,self.rng.uniform(size=self.y.shape),.05,profile,u1)
                lower=endpoint_score(self.y,self.p,self.s)
                upper=robust_score(estimate,self.p,self.s,.05,u1)
                self.assertTrue(np.all(lower<=upper+1e-12))
                for alpha in (.01,.05,.2): self.assertLessEqual(quantile(lower,alpha),quantile(upper,alpha)+1e-12)
    def test_zero_error_equivalence(self):
        np.testing.assert_array_equal(robust_score(self.y,self.p,self.s,0),endpoint_score(self.y,self.p,self.s))
    def test_joint_grid_is_same_statistical_method(self):
        np.testing.assert_allclose(endpoint_score(self.y,self.p,self.s),joint_grid_score(self.y,self.p,self.s),rtol=1e-14,atol=1e-14)
    def test_worst_case_allowance_matches_error_box(self):
        for v in (0,.3,.75,1,2.5,3):
            errors=np.array(list(itertools.product((-.05,.05),repeat=2)))
            coeff=reconstruct(errors)
            self.assertAlmostEqual(max(coeff[:,0]+v*coeff[:,1]),error_allowance(v,.05),places=12)
if __name__=='__main__':unittest.main()
