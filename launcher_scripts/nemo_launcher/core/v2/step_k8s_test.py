from hera.workflows import (
    Workflow,
    Resource,
)
from nemo_launcher.core.v2.step_k8s import (
    prune_tree,
    create_pytorchjob_resource,
    create_mpijob_resource,
    BACKOFF_LIMIT,
)
import pytest
import yaml


@pytest.mark.parametrize(
    "in_tree,expected_out_tree",
    [
        ({4: 3}, {4: 3}),
        ({4: {3: None}, 5: 6}, {5: 6}),
        ({4: None, 1: [{2: {}}, {7: [{}, {}]}, 6, None, ""]}, {1: [6, ""]}),
    ],
)
def test_prune_tree(in_tree, expected_out_tree):
    out_tree = prune_tree(in_tree)
    assert out_tree == expected_out_tree


def test_pytorchjob_in_workflow():
    with Workflow(generate_name="test-create-", entrypoint="first-step-",) as w:
        create_pytorchjob_resource(
            generate_name="first-step-",
            namespace=w.namespace,
            image="python:3.10",
            n_workers=2,
            gpus_per_worker=1,
        )

    expected_workflow_manifest = {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "Workflow",
        "metadata": {"generateName": "test-create-"},
        "spec": {
            "entrypoint": "first-step-",
            "templates": [
                {
                    "name": "first-step-",
                    "outputs": {
                        "parameters": [
                            {
                                "name": "metadata_name",
                                "valueFrom": {"jsonPath": "{.metadata.name}",},
                            },
                            {
                                "name": "metadata_namespace",
                                "valueFrom": {"jsonPath": "{.metadata.namespace}",},
                            },
                        ],
                    },
                    "resource": {
                        "action": "create",
                        # Check manifest separately
                        # "manifest": "apiVersion: kubeflow.org/v1\nkind: PyTorchJob..."
                        "setOwnerReference": True,
                        "successCondition": "status.replicaStatuses.Worker.succeeded = 2",
                    },
                }
            ],
        },
    }
    expected_pytorchjob_manifest = {
        "apiVersion": "kubeflow.org/v1",
        "kind": "PyTorchJob",
        "metadata": {"generateName": "first-step-",},
        "spec": {
            "elasticPolicy": {
                "maxReplicas": 2,
                "maxRestarts": 10,
                "minReplicas": 2,
                "nProcPerNode": 1,
                "rdzvBackend": "c10d",
            },
            "pytorchReplicaSpecs": {
                "Worker": {
                    "replicas": 2,
                    "restartPolicy": "OnFailure",
                    "template": {
                        "metadata": {
                            "annotations": {"sidecar.istio.io/inject": "false",},
                            "labels": {
                                "workflows.argoproj.io/workflow": "{{workflow.name}}",
                            },
                        },
                        "spec": {
                            "containers": [
                                {
                                    "image": "python:3.10",
                                    "name": "pytorch",
                                    "resources": {"limits": {"nvidia.com/gpu": "1",},},
                                },
                            ],
                        },
                    },
                },
            },
            "runPolicy": {"cleanPodPolicy": "None", "backoffLimit": BACKOFF_LIMIT},
        },
    }

    workflow_obj = w.to_dict()
    pytorch_spec: str = workflow_obj["spec"]["templates"][0]["resource"].pop("manifest")
    assert workflow_obj == expected_workflow_manifest

    pytorch_spec = yaml.safe_load(pytorch_spec)

    assert pytorch_spec == expected_pytorchjob_manifest


def test_mpijob_in_workflow():
    with Workflow(generate_name="test-create-", entrypoint="first-step-",) as w:
        create_mpijob_resource(
            generate_name="first-step-",
            namespace=w.namespace,
            image="python:3.10",
            n_workers=2,
        )

    expected_workflow_manifest = {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "Workflow",
        "metadata": {"generateName": "test-create-"},
        "spec": {
            "entrypoint": "first-step-",
            "templates": [
                {
                    "name": "first-step-",
                    "outputs": {
                        "parameters": [
                            {
                                "name": "metadata_name",
                                "valueFrom": {"jsonPath": "{.metadata.name}",},
                            },
                            {
                                "name": "metadata_namespace",
                                "valueFrom": {"jsonPath": "{.metadata.namespace}",},
                            },
                        ],
                    },
                    "resource": {
                        "action": "create",
                        # Check manifest separately
                        # "manifest": "apiVersion: kubeflow.org/v1\nkind: MPIJob..."
                        "setOwnerReference": True,
                        "successCondition": "status.replicaStatuses.Launcher.succeeded = 1",
                    },
                }
            ],
        },
    }
    expected_mpijob_manifest = {
        "apiVersion": "kubeflow.org/v1",
        "kind": "MPIJob",
        "metadata": {"generateName": "first-step-",},
        "spec": {
            "mpiReplicaSpecs": {
                "Launcher": {
                    "replicas": 1,
                    "restartPolicy": "OnFailure",
                    "template": {
                        "metadata": {
                            "annotations": {"sidecar.istio.io/inject": "false",},
                            "labels": {
                                "workflows.argoproj.io/workflow": "{{workflow.name}}",
                            },
                        },
                        "spec": {
                            "containers": [
                                {"image": "python:3.10", "name": "mpi-launcher",},
                            ],
                        },
                    },
                },
                "Worker": {
                    "replicas": 2,
                    "restartPolicy": "OnFailure",
                    "template": {
                        "metadata": {
                            "annotations": {"sidecar.istio.io/inject": "false",},
                            "labels": {
                                "workflows.argoproj.io/workflow": "{{workflow.name}}",
                            },
                        },
                        "spec": {
                            "containers": [
                                {
                                    "args": ["-De",],
                                    "command": ["/usr/sbin/sshd",],
                                    "image": "python:3.10",
                                    "name": "mpi-worker",
                                },
                            ],
                        },
                    },
                },
            },
            "runPolicy": {"cleanPodPolicy": "Running", "backoffLimit": BACKOFF_LIMIT},
        },
    }

    workflow_obj = w.to_dict()
    mpijob_spec: str = workflow_obj["spec"]["templates"][0]["resource"].pop("manifest")
    assert workflow_obj == expected_workflow_manifest

    mpijob_spec = yaml.safe_load(mpijob_spec)

    assert mpijob_spec == expected_mpijob_manifest
