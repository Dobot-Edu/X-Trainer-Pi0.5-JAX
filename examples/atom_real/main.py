import dataclasses
import logging

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    prompt: str = "pick up the object"
    action_horizon: int = 25
    control_hz: float = 20.0

    num_episodes: int = 1
    max_episode_steps: int = 1000

    camera_top_serial: str = ""
    camera_top_id: int = -1
    camera_top_width: int = 1280
    camera_top_height: int = 720
    camera_top_fps: float = 60.0
    camera_left_wrist_serial: str = ""
    camera_right_wrist_serial: str = ""
    camera_fps: float = 30.0

    render_height: int = 224
    render_width: int = 224

    max_joint_delta: float = 0.17
    ramp_step: float = 0.01
    ramp_max_steps: int = 100

    enable_upper_control_on_connect: bool = True
    disable_upper_control_on_close: bool = True


def main(args: Args) -> None:
    from examples.atom_real import env as _env

    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    metadata = ws_client_policy.get_server_metadata()
    logging.info("Server metadata: %s", metadata)

    environment = _env.AtomRealEnvironment(
        camera_top_serial=args.camera_top_serial,
        camera_top_id=args.camera_top_id,
        camera_top_width=args.camera_top_width,
        camera_top_height=args.camera_top_height,
        camera_top_fps=args.camera_top_fps,
        camera_left_wrist_serial=args.camera_left_wrist_serial,
        camera_right_wrist_serial=args.camera_right_wrist_serial,
        camera_fps=args.camera_fps,
        render_height=args.render_height,
        render_width=args.render_width,
        prompt=args.prompt,
        reset_pose=metadata.get("reset_pose"),
        max_joint_delta=args.max_joint_delta,
        ramp_step=args.ramp_step,
        ramp_max_steps=args.ramp_max_steps,
        enable_upper_control_on_connect=args.enable_upper_control_on_connect,
        disable_upper_control_on_close=args.disable_upper_control_on_close,
    )

    runtime = _runtime.Runtime(
        environment=environment,
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[],
        max_hz=args.control_hz,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    try:
        runtime.run()
    finally:
        environment.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
