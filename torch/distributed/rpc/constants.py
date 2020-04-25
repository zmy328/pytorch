from torch.distributed.constants import default_pg_timeout

from . import (
    _DEFAULT_RPC_TIMEOUT_SEC,
    _UNSET_RPC_TIMEOUT,
    _DEFAULT_INIT_METHOD,
    _DEFAULT_NUM_SEND_RECV_THREADS
)

# For any RpcAgent.
DEFAULT_RPC_TIMEOUT_SEC = _DEFAULT_RPC_TIMEOUT_SEC
DEFAULT_INIT_METHOD = _DEFAULT_INIT_METHOD

# For ProcessGroupAgent.
DEFAULT_NUM_SEND_RECV_THREADS = _DEFAULT_NUM_SEND_RECV_THREADS
# Same default timeout as in c10d.
DEFAULT_PROCESS_GROUP_TIMEOUT = default_pg_timeout
# Value indicating that timeout is not set for RPC call, and the default should be used.
UNSET_RPC_TIMEOUT = _UNSET_RPC_TIMEOUT
