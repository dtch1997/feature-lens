import unittest
import pytest
from unittest.mock import Mock, patch
import pathlib
from typing import Any, List

from feature_lens.core.experiment import (
    ExperimentArtifact, ExperimentStage, ExperimentSpec,
    ExperimentState, ExperimentRunner, FileSystemExperimentState,
    ArtifactSpec, _validate_artifacts_against_specs
)

class TestExperimentArtifact(unittest.TestCase):
    def test_artifact_creation(self):
        artifact = ExperimentArtifact("test_artifact", "test_data")
        self.assertEqual(artifact.name, "test_artifact")
        self.assertEqual(artifact.data, "test_data")

    def test_artifact_spec(self):
        artifact = ExperimentArtifact("test_artifact", "test_data")
        spec = artifact.spec()
        self.assertEqual(spec, ("test_artifact", ExperimentArtifact))

class TestExperimentStage(unittest.TestCase):
    def setUp(self):
        self.input_spec = [("input1", ExperimentArtifact)]
        self.output_spec = [("output1", ExperimentArtifact)]
        
        class ConcreteStage(ExperimentStage):
            def run(self, input_artifacts):
                return [ExperimentArtifact("output1", "processed_data")]
        
        self.stage = ConcreteStage("test_stage", self.input_spec, self.output_spec)

    def test_stage_creation(self):
        self.assertEqual(self.stage.name, "test_stage")
        self.assertEqual(self.stage.input_spec, self.input_spec)
        self.assertEqual(self.stage.output_spec, self.output_spec)

    def test_stage_call(self):
        input_artifact = ExperimentArtifact("input1", "test_data")
        outputs = self.stage([input_artifact])
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].name, "output1")
        self.assertEqual(outputs[0].data, "processed_data")

    def test_stage_call_invalid_input(self):
        with self.assertRaises(ValueError):
            self.stage([ExperimentArtifact("invalid_input", "test_data")])

class TestExperimentSpec(unittest.TestCase):
    def test_spec_creation(self):
        stage = Mock(spec=ExperimentStage)
        spec = ExperimentSpec("test_experiment", [stage], "Test description")
        self.assertEqual(spec.name, "test_experiment")
        self.assertEqual(spec.stages, [stage])
        self.assertEqual(spec.description, "Test description")

class TestExperimentRunner(unittest.TestCase):
    def setUp(self):

        a1 = ExperimentArtifact("artifact1", "data1")
        a2 = ExperimentArtifact("artifact2", "data2")

        mock_stage1 = Mock(spec=ExperimentStage)
        mock_stage1.name = "stage1"
        mock_stage1.input_spec = []
        mock_stage1.output_spec = [a1.spec(), a2.spec()]
        mock_stage1.run.return_value = [a1, a2]

        mock_stage2 = Mock(spec=ExperimentStage)
        mock_stage2.name = "stage2"
        mock_stage2.input_spec = [a1.spec(), a2.spec()]
        mock_stage2.output_spec = []
        mock_stage2.run.return_value = []

        self.mock_spec = Mock(spec=ExperimentSpec)
        self.mock_spec.stages = [mock_stage1, mock_stage2]

        self.mock_state = Mock(spec=ExperimentState)
        self.mock_state.is_stage_complete.return_value = False
        self.runner = ExperimentRunner(self.mock_spec, self.mock_state)

    def test_run(self):
        self.runner.run()
        
        self.mock_state.is_stage_complete.assert_any_call("stage1")
        self.mock_state.is_stage_complete.assert_any_call("stage2")
        self.mock_state.read_artifacts.assert_called()
        self.mock_state.write_artifacts.assert_called()

class TestFileSystemExperimentState(unittest.TestCase):
    def setUp(self):
        self.temp_dir = pathlib.Path("temp_test_dir")
        self.temp_dir.mkdir(exist_ok=True)
        self.state = FileSystemExperimentState(self.temp_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch('feature_lens.core.experiment.ExperimentArtifact')
    def test_read_write_artifact(self, MockArtifact):
        mock_artifact = Mock(spec=ExperimentArtifact)
        mock_artifact.name = "test_artifact"
        mock_artifact.data = "test_data"
        mock_artifact.to_bytes.return_value = b"test_data"
        mock_artifact.spec.return_value = ArtifactSpec("test_artifact", ExperimentArtifact)
        
        self.state.write_artifact(mock_artifact)
        
        artifact_path = self.temp_dir / "artifacts" / "test_artifact.pkl"
        self.assertTrue(artifact_path.exists())
        
        MockArtifact.from_bytes.return_value = mock_artifact
        read_artifact = self.state.read_artifact(ArtifactSpec("test_artifact", MockArtifact))
        
        self.assertEqual(read_artifact, mock_artifact)

def test_validate_artifacts_against_specs():
    class SubArtifact(ExperimentArtifact):
        pass

    valid_artifacts = [
        ExperimentArtifact("artifact1", "data1"),
        SubArtifact("artifact2", "data2")
    ]
    valid_specs = [
        ("artifact1", ExperimentArtifact),
        ("artifact2", ExperimentArtifact)
    ]

    # This should not raise an exception
    _validate_artifacts_against_specs(valid_artifacts, valid_specs)

    invalid_artifacts = [
        ExperimentArtifact("artifact1", "data1"),
        ExperimentArtifact("artifact3", "data3")
    ]

    with pytest.raises(ValueError):
        _validate_artifacts_against_specs(invalid_artifacts, valid_specs)

    invalid_type_artifacts = [
        ExperimentArtifact("artifact1", "data1"),
        ExperimentArtifact("artifact2", "data2")
    ]
    invalid_type_specs = [
        ("artifact1", ExperimentArtifact),
        ("artifact2", SubArtifact)
    ]

    with pytest.raises(TypeError):
        _validate_artifacts_against_specs(invalid_type_artifacts, invalid_type_specs)

if __name__ == '__main__':
    unittest.main()