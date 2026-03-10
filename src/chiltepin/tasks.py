# SPDX-License-Identifier: Apache-2.0

"""Task decorators for Chiltepin workflows.

This module provides decorators for defining workflow tasks that can be executed
on configured resources. Tasks are the fundamental units of work in Chiltepin workflows.

Available Decorators
--------------------
- :func:`python_task`: Execute Python functions as workflow tasks
- :func:`bash_task`: Execute shell commands as workflow tasks
- :func:`join_task`: Coordinate multiple tasks without blocking workflow execution

For comprehensive usage examples and best practices, see the :doc:`tasks` documentation.

Examples
--------
Define a simple Python task::

    from chiltepin.tasks import python_task

    @python_task
    def add_numbers(a, b):
        return a + b

    # Execute on a specific resource
    result = add_numbers(5, 3, executor=["compute"]).result()

Define a bash task::

    from chiltepin.tasks import bash_task

    @bash_task
    def list_files(directory):
        return f"ls -la {directory}"

    # Returns exit code (0 = success)
    exit_code = list_files("/tmp", executor=["compute"]).result()

Define an MPI task using task geometry::

    @bash_task
    def run_mpi_simulation(input_file):
        return f"$PARSL_MPI_PREFIX ./simulation {input_file}"

    # Specify parallel resource requirements
    exit_code = run_mpi_simulation(
        "config.in",
        executor=["mpi"],
        chiltepin_task_geometry={
            "num_nodes": 4,
            "num_ranks": 16,
            "ranks_per_node": 4
        }
    ).result()
"""

from functools import wraps
from inspect import Parameter, signature
from typing import Callable, Optional

from parsl.app.app import bash_app, join_app, python_app


def _create_filtered_wrapper(function: Callable) -> Callable:
    """Create a wrapper that filters kwargs to only pass what the function accepts.

    This helper function creates an intermediate wrapper for Parsl app decorators.
    The wrapper accepts any arguments that Parsl injects (stdout, stderr, etc.)
    but only forwards the ones that the user's function signature expects.

    Parameters
    ----------
    function: Callable
        The user's function to wrap

    Returns
    -------
    Callable
        A wrapper function that filters kwargs based on the function's signature
    """
    sig = signature(function)
    func_params = sig.parameters

    # Check if the function accepts **kwargs (VAR_KEYWORD)
    has_var_keyword = any(
        param.kind == Parameter.VAR_KEYWORD for param in func_params.values()
    )

    @wraps(function)
    def wrapper(*args, **kwargs):
        # If function has **kwargs, pass all kwargs through
        # Otherwise, filter to only include parameters the function explicitly accepts
        if has_var_keyword:
            return function(*args, **kwargs)
        else:
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in func_params}
            return function(*args, **filtered_kwargs)

    return wrapper


def _merge_chiltepin_task_geometry(
    chiltepin_task_geometry: Optional[dict], kwargs: dict
) -> None:
    """Merge chiltepin_task_geometry into parsl_resource_specification in kwargs.

    This helper function handles the merging of chiltepin_task_geometry into the
    parsl_resource_specification parameter. If both are provided, they are merged
    with chiltepin_task_geometry taking precedence for overlapping keys. If
    parsl_resource_specification is None, it is treated as an empty dict.

    Parameters
    ----------
    chiltepin_task_geometry: dict or None
        Task geometry specification to merge
    kwargs: dict
        Keyword arguments dictionary to modify in place

    Returns
    -------
    None
        Modifies kwargs in place

    Raises
    ------
    TypeError
        If chiltepin_task_geometry is not None and not a dict
    """
    if chiltepin_task_geometry is not None:
        # Validate that chiltepin_task_geometry is a dict
        if not isinstance(chiltepin_task_geometry, dict):
            raise TypeError(
                f"chiltepin_task_geometry must be a dict, got {type(chiltepin_task_geometry).__name__}"
            )

        # Make a defensive copy to prevent mutations
        geometry_copy = dict(chiltepin_task_geometry)

        if "parsl_resource_specification" in kwargs:
            existing_spec = kwargs["parsl_resource_specification"]
            # Treat None as an empty dict
            if existing_spec is None:
                existing_spec = {}
            # Merge: start with existing spec, update with geometry
            merged_spec = dict(existing_spec)
            merged_spec.update(geometry_copy)
            kwargs["parsl_resource_specification"] = merged_spec
        else:
            kwargs["parsl_resource_specification"] = geometry_copy


class MethodWrapper:
    """Wrapper that preserves method behavior for decorated functions.

    This descriptor ensures that when a decorated function is accessed as a
    method, it properly creates a bound method with the instance.
    """

    def __init__(self, func, wrapper_func):
        self.func = func
        self.wrapper_func = wrapper_func
        # Copy over metadata
        wraps(func)(self)

    def __get__(self, obj, objtype=None):
        """Support instance methods."""
        if obj is None:
            # Accessed on class, return self
            return self
        # Return a bound version of the wrapper
        from functools import partial

        return partial(self.wrapper_func, obj)

    def __call__(self, *args, **kwargs):
        """Support standalone function calls."""
        return self.wrapper_func(*args, **kwargs)


def python_task(function: Callable) -> Callable:
    """Decorator function for making Chiltepin python tasks.

    The decorator transforms the function into a Parsl python_app but adds an executor
    argument such that the executor for the function can be chosen dynamically at runtime.

    Parameters
    ----------
    function: Callable
        The function to be decorated to yield a Python workflow task. This function can be a
        stand-alone function or a class method. If it is a class method, it can make use of
        `self` to access object state.

    Other Parameters
    ----------------
    The decorated function includes the following additional parameters at call time:

    executor: str or list of str, default="all"
        Resource name(s) where the task should execute. Can be a single resource name or
        a list of resource names. Defaults to "all" which allows execution on any configured
        resource.

    chiltepin_task_geometry: dict, optional
        Specification of parallel task geometry for MPI applications. This parameter is
        mapped to Parsl's ``parsl_resource_specification``. The dictionary should contain:

        - **num_nodes** (int): Number of nodes required for the task
        - **num_ranks** (int): Total number of MPI ranks
        - **ranks_per_node** (int): Number of MPI ranks per node

        Example::

            chiltepin_task_geometry={
                "num_nodes": 4,
                "num_ranks": 16,
                "ranks_per_node": 4
            }

    inputs: list of AppFuture, optional
        List of futures that must complete before this task starts. Used to create
        task dependencies without passing data between tasks. See Parsl's documentation
        for details.

    .. note::
       All keyword arguments supported by Parsl's ``python_app`` decorator (such as
       ``outputs``, ``walltime``, etc.) are also accepted and passed through
       to the underlying Parsl app.

    Returns
    -------
    Callable
        The decorated function that can be called as a workflow task.

    Examples
    --------
    Basic usage::

        @python_task
        def compute(x):
            return x ** 2

        result = compute(5, executor=["compute"]).result()

    MPI task with task geometry::

        @python_task
        def run_mpi_code(params):
            # MPI code execution
            return "result"

        future = run_mpi_code(
            params,
            executor=["mpi"],
            chiltepin_task_geometry={"num_nodes": 2, "num_ranks": 8, "ranks_per_node": 4}
        )

    """

    def function_wrapper(
        *args,
        executor="all",
        chiltepin_task_geometry=None,
        **kwargs,
    ):
        # Map chiltepin_task_geometry to Parsl's parsl_resource_specification
        _merge_chiltepin_task_geometry(chiltepin_task_geometry, kwargs)

        return python_app(_create_filtered_wrapper(function), executors=executor)(
            *args, **kwargs
        )

    return MethodWrapper(function, function_wrapper)


def bash_task(function: Callable) -> Callable:
    """Decorator function for making Chiltepin bash tasks.

    The decorator transforms the function into a Parsl bash_app but adds an executor
    argument such that the executor for the function can be chosen dynamically at runtime.

    Parameters
    ----------
    function: Callable
        The function to be decorated to yield a Bash workflow task. This function can be a
        stand-alone function or a class method. If it is a class method, it can make use of
        `self` to access object state. The function must return a string that contains a
        series of bash commands to be executed.

    Other Parameters
    ----------------
    The decorated function includes the following additional parameters at call time:

    executor: str or list of str, default="all"
        Resource name(s) where the task should execute. Can be a single resource name or
        a list of resource names. Defaults to "all" which allows execution on any configured
        resource.

    chiltepin_task_geometry: dict, optional
        Specification of parallel task geometry for MPI applications. This parameter is
        mapped to Parsl's ``parsl_resource_specification``. The dictionary should contain:

        - **num_nodes** (int): Number of nodes required for the task
        - **num_ranks** (int): Total number of MPI ranks
        - **ranks_per_node** (int): Number of MPI ranks per node

        Example::

            chiltepin_task_geometry={
                "num_nodes": 4,
                "num_ranks": 16,
                "ranks_per_node": 4
            }

    stdout: str or tuple, optional
        File path for capturing standard output. Can be a string path or a tuple of
        (path, mode) where mode is typically 'w' for write or 'a' for append.

    stderr: str or tuple, optional
        File path for capturing standard error. Can be a string path or a tuple of
        (path, mode) where mode is typically 'w' for write or 'a' for append.

    inputs: list of AppFuture, optional
        List of futures that must complete before this task starts. Used to create
        task dependencies without passing data between tasks. See Parsl's documentation
        for details.

    .. note::
       All keyword arguments supported by Parsl's ``bash_app`` decorator (such as
       ``outputs``, ``walltime``, etc.) are also accepted and passed through
       to the underlying Parsl app.

    Returns
    -------
    Callable
        The decorated function that can be called as a workflow task. Returns the exit
        code of the bash command (0 indicates success).

    Examples
    --------
    Basic usage::

        @bash_task
        def compile_code():
            return "gcc -o program program.c"

        exit_code = compile_code(executor=["compute"]).result()

    MPI task with task geometry::

        @bash_task
        def run_mpi_simulation(input_file):
            return f"$PARSL_MPI_PREFIX ./simulation {input_file}"

        exit_code = run_mpi_simulation(
            "config.in",
            executor=["mpi"],
            chiltepin_task_geometry={"num_nodes": 4, "num_ranks": 16, "ranks_per_node": 4},
            stdout="output.log"
        ).result()

    """

    def function_wrapper(
        *args,
        executor="all",
        chiltepin_task_geometry=None,
        **kwargs,
    ):
        # Map chiltepin_task_geometry to Parsl's parsl_resource_specification
        _merge_chiltepin_task_geometry(chiltepin_task_geometry, kwargs)

        return bash_app(_create_filtered_wrapper(function), executors=executor)(
            *args, **kwargs
        )

    return MethodWrapper(function, function_wrapper)


def join_task(function: Callable) -> Callable:
    """Decorator function for making Chiltepin join tasks.

    The decorator transforms the function into a Parsl join_app. A parsl @join_app decorator
    accomplishes the same thing.  This decorator is added to provide API consistency so that
    users can use @join_task rather than @join_app along with @python_task and @bash_task.

    Parameters
    ----------

    function: Callable
        The function to be decorated to yield a join workflow task. This function can be a
        stand-alone function or a class method. If it is a class method, it can make use of
        `self` to access object state. The function is expected to call multiple python or
        bash tasks and return a Future that encapsulates the result of those tasks.


    Returns
    -------

    Callable

    """

    def function_wrapper(
        *args,
        **kwargs,
    ):
        return join_app(_create_filtered_wrapper(function))(*args, **kwargs)

    return MethodWrapper(function, function_wrapper)
