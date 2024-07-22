""" Utilities for defining and running an experiment reproducibly. """
from __future__ import annotations

import abc
import pathlib
from typing import Any, Type, NamedTuple
from dataclasses import dataclass, field
from feature_lens.core.patterns import AbstractDataclass, ByteSerializable

def list_field():
    """ Helper function to create a default list field. """
    return field(default_factory=list)

@dataclass(frozen=True)
class ExperimentArtifact(ByteSerializable):
    """ Abstract class for defining an experiment artifact. 
    
    Subclasses should override the 'data' field type declaration.
    """
    name: str
    data: Any

    def spec(self) -> ArtifactSpec:
        return (self.name, type(self))

def _validate_artifacts_against_specs(
    artifacts: list[ExperimentArtifact], 
    artifact_specs: list[ArtifactSpec]
):
    received_specs = {a.name: type(a) for a in artifacts}
    expected_specs = {name: cls for name, cls in artifact_specs}
    for name, cls in expected_specs.items():
        if name not in received_specs:
            raise ValueError(f"Expected artifact {name} not found.")
        if not issubclass(received_specs[name], cls):
            raise TypeError(f"Expected artifact {name} to be subclass of {cls} but got {received_specs[name]}")

class ArtifactSpec(NamedTuple):
    name: str
    cls: Type[ExperimentArtifact]

@dataclass(frozen=True)
class ExperimentStage(ByteSerializable):
    """ Abstract class for defining an experiment stage. """
    name: str
    input_spec: list[ArtifactSpec]
    output_spec: list[ArtifactSpec]

    def __call__(self, inputs: list[ExperimentArtifact]) -> list[ExperimentArtifact]:        
        _validate_artifacts_against_specs(inputs, self.input_spec)
        outputs = self.run(inputs)
        _validate_artifacts_against_specs(outputs, self.output_spec)
        return outputs

    @abc.abstractmethod
    def run(self, input_artifacts: list[ExperimentArtifact]) -> list[ExperimentArtifact]:
        raise NotImplementedError
    
@dataclass(frozen=True)
class ExperimentSpec(ByteSerializable):
    name: str
    stages: list[ExperimentStage]
    description: str = ""

class ExperimentState(abc.ABC):
    
    @abc.abstractmethod
    def is_stage_complete(self, stage_name):
        pass

    @abc.abstractmethod
    def read_artifact(self, artifact_spec: ArtifactSpec):
        pass

    @abc.abstractmethod
    def write_artifact(self, artifact: ExperimentArtifact):
        pass

    def read_artifacts(self, artifact_specs: list[ArtifactSpec]):
        return [self.read_artifact(spec) for spec in artifact_specs]
    
    def write_artifacts(self, artifacts: list[ExperimentArtifact]):
        for artifact in artifacts:
            self.write_artifact(artifact)

class ExperimentRunner:

    spec: ExperimentSpec
    state: ExperimentState

    def __init__(self, spec: ExperimentSpec, state: ExperimentState):
        self.spec = spec
        self.state = state

    def run(self):
        for stage in self.spec.stages:
            if not self.state.is_stage_complete(stage.name):
                self._run_stage(stage)
            else:
                print(f"Skipping completed stage: {stage.name}")

    def _run_stage(self, stage: ExperimentStage):
        inputs = self.state.read_artifacts(stage.input_spec)
        outputs = stage(inputs)
        self.state.write_artifacts(outputs)

class FileSystemExperimentState(ExperimentState):
    def __init__(self, root_dir: pathlib.Path):
        self.root_dir = root_dir
        self.root_dir.mkdir(exist_ok=True)

        # make artifacts dir
        self.artifacts_dir = self.root_dir / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)

    def get_artifact_path(self, artifact_name: str) -> pathlib.Path:
        return (self.artifacts_dir / artifact_name).with_suffix(".pkl")

    def is_stage_complete(self, stage: ExperimentStage):
        # Check all expected outputs exist
        for spec in stage.output_spec:
            if not self.get_artifact_path(spec.name).exists():
                return False
        return True
    
    def read_artifact(self, artifact_spec: ArtifactSpec):
        path = self.get_artifact_path(artifact_spec.name)
        with open(path, "rb") as f:
            bytes = f.read()
            return artifact_spec.cls.from_bytes(bytes)
    
    def write_artifact(self, artifact: ExperimentArtifact):
        path = self.get_artifact_path(artifact.spec().name)
        with open(path, "wb") as f:
            f.write(artifact.to_bytes())