from openenv.core.env_server.http_server import create_app

from models import PacketAction, PacketObservation
from server.pkt_schd_rl_environment import PacketSchedEnv

app = create_app(
    PacketSchedEnv,
    PacketAction,
    PacketObservation,
    env_name="packet_scheduling",
    max_concurrent_envs=1,
)