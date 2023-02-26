from esbcpo.algos.policy_gradient import PG
from esbcpo.algos.trpo import TRPO
from esbcpo.algos.esb_cpo import ESB_CPO


REGISTRY = {
    'pg': PG,
    'trpo': TRPO,
    'esb_cpo': ESB_CPO,
}
