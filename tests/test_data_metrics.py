import numpy as np
import pytest
from task99.data import make_split, mask_faces, make_resamples, roll_seed, prepare
from task99.metrics import bias_variance, disagreement


def test_split_and_resampling_preserve_identity_and_exclude_holdouts():
    y = np.repeat(np.arange(40), 10)
    split = make_split(y, 99)
    assert [len(split[k]) for k in ['train','validation','test']] == [240,80,80]
    assert len(set(np.concatenate(list(split.values())))) == 400
    for k, count in [('train',6),('validation',2),('test',2)]:
        assert np.array_equal(np.bincount(y[split[k]]), np.full(40,count))
    samples = make_resamples(y,split['train'],99)
    assert samples.shape == (30,240)
    assert set(samples.ravel()) <= set(split['train'])
    assert all(np.array_equal(np.bincount(y[s]),np.full(40,6)) for s in samples)


def test_mask_changes_exactly_center_and_preserves_input():
    original = np.ones((2,4096))
    masked = mask_faces(original).reshape(2,64,64)
    assert np.count_nonzero(masked == 0) == 2*256
    assert np.all(masked[:,24:40,24:40] == 0)
    assert np.all(original == 1)


def test_bias_variance_known_vectors_and_disagreement():
    p=np.array([[[1.,0.]],[[0.,1.]]])
    result=bias_variance(p,np.array([0]))
    assert result['bias_squared'] == pytest.approx(.5)
    assert result['variance'] == pytest.approx(.5)
    assert result['brier_mse'] == pytest.approx(1.)
    assert disagreement(np.array([[0,0],[0,1],[1,1]])) == pytest.approx(2/3)
    assert disagreement(np.array([[0,1]])) is None


def test_roll_number_uses_all_digits_in_order():
    assert roll_seed('2026-CS-0012') == 20260012
    with pytest.raises(ValueError):
        roll_seed('no digits')


def test_saved_split_refuses_seed_change_and_real_roll_overrides_provisional(tmp_path):
    data=tmp_path/'data'
    data.mkdir()
    np.savez_compressed(data/'olivetti.npz',X=np.zeros((400,4096),dtype=np.float32),y=np.repeat(np.arange(40),10))
    config={'seed':99,'roll_number':'CS-00123','resamples':30}
    _,_,first,_,seed=prepare(tmp_path,config)
    assert seed==123
    before=(data/'split.npz').read_bytes()
    _,_,second,_,_=prepare(tmp_path,config)
    assert all(np.array_equal(first[k],second[k]) for k in first)
    with pytest.raises(ValueError,match='another seed'):
        prepare(tmp_path,dict(config,roll_number='CS-124'))
    assert (data/'split.npz').read_bytes()==before
