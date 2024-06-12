import datetime
import json
import os
import uuid
import numpy as np
import cattrs
from attrs import define, field
import logging
import tempfile

from settings import Settings

logger = logging.getLogger(__name__)

@define
class Session:
    """A class to store the all information about a session, including the chat
    history and the current scene."""

    session_id: str
    start_time: str
    scene: str
    chat_history_for_display: list[tuple]
    chat_counter: int
    output_dir: str

    @classmethod
    def create(cls):
        return cls.create_for_scene(Settings.default_scene)

    @classmethod
    def create_for_scene(cls, scene: str):
        output_dir = tempfile.mkdtemp()  # Create a temporary directory
        session = cls(
            session_id=str(uuid.uuid4()),
            start_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            scene=scene,
            chat_history_for_display=[(None, Settings.INITIAL_MSG_FOR_DISPLAY)],
            chat_counter=0,
            output_dir=output_dir
        )
        logger.info(
            f"Creating a new session {session.session_id} with scene {session.scene} and output dir {session.output_dir}."
        )
        return session

    def convert_float32(self, obj):
        """Convert all np.float32 values in the given object to Python float."""
        if isinstance(obj, np.float32):
            return float(obj)

        if isinstance(obj, list):
            return [self.convert_float32(item) for item in obj]

        if isinstance(obj, tuple):
            return tuple(self.convert_float32(item) for item in obj)

        if isinstance(obj, dict):
            return {key: self.convert_float32(value) for key, value in obj.items()}

        return obj
    
    def get_session_output_dir(self):
        # Create the directory and any parent directories if they don't exist
        os.makedirs(os.path.join(self.output_dir, self.scene, self.session_id), exist_ok=True)
        return os.path.join(self.output_dir, self.scene, self.session_id)

    def save(self) -> None:
        """Save the session as a json file."""
        logger.info(f"Saving session {self.session_id} to disk.")

        structured_data = cattrs.unstructure(self)
        structured_data.pop("chat_history_for_display", None)
        # Convert all np.float32 to float
        converted_data = self.convert_float32(structured_data)

        with open(
            os.path.join(
                self.get_session_output_dir(), f"{self.session_id}.json"
            ),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(converted_data, file, indent=4)
        logger.info(f"Session {self.session_id} saved to {self.get_session_output_dir()}.")
