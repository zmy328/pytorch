#!/usr/bin/python3
import importlib
import inspect
import os

import torch
from src.ATen.code_template import CodeTemplate
from torch.distributed.nn.jit.templates import dir_path as TEMPLATE_DIR_PATH


def get_return_type_from_callable(callable_obj):
    sig = inspect.signature(callable_obj)
    return_annotation = sig.return_annotation
    if return_annotation is inspect.Signature.empty:
        return callable_obj
    return return_annotation


def get_arg_return_types_from_interface(module_interface):
    assert getattr(
        module_interface, "__torch_script_interface__", False
    ), "Expect a class decorated by @torch.jit.interface."
    qualified_name = torch.jit._qualified_name(module_interface)
    cu = torch.jit._python_cu
    module_interface_c = cu.get_interface(qualified_name)
    assert (
        "forward" in module_interface_c.getMethodNames()
    ), "Expect forward in interface methods, while it has {}".format(
        module_interface_c.getMethodNames()
    )
    method_schema = module_interface_c.getMethod("forward")

    arg_str_list = []
    arg_type_str_list = []
    for argument in method_schema.arguments:
        arg_str_list.append(argument.name)

        if argument.has_default_value():
            default_value_str = " = {}".format(argument.default)
        else:
            default_value_str = ""
        arg_type_str = "{name}: {type}{default_value}".format(
            name=argument.name, type=argument.type, default_value=default_value_str
        )
        arg_type_str_list.append(arg_type_str)

    arg_str_list = arg_str_list[1:]  # Remove "self".
    args_str = ", ".join(arg_str_list)

    arg_type_str_list = arg_type_str_list[1:]  # Remove "self".
    arg_types_str = ", ".join(arg_type_str_list)

    assert len(method_schema.returns) == 1
    argument = method_schema.returns[0]
    return_type_str = str(argument.type)

    return args_str, arg_types_str, return_type_str


def write(out_path, text):
    try:
        with open(out_path, "r") as f:
            old_text = f.read()
    except IOError:
        old_text = None
    if old_text != text:
        with open(out_path, "w") as f:
            print("Writing {}".format(out_path))
            f.write(text)
    else:
        print("Skipped writing {}".format(out_path))


def instantiate_remote_module_template(
    generated_module_name, module_interface_cls, is_scriptable
):
    if is_scriptable:
        args_str, arg_types_str, return_type_str = get_arg_return_types_from_interface(
            module_interface_cls
        )
        kwargs_str = ""
        arrow_and_return_type_str = f" -> {return_type_str}"
        arrow_and_future_return_type_str = f" -> Future[{return_type_str}]"
    else:
        args_str = "*args"
        kwargs_str = "**kwargs"
        arg_types_str = "*args, **kwargs"
        arrow_and_return_type_str = ""
        arrow_and_future_return_type_str = ""

    jit_decorator_str_map = dict(
        jit_script_decorator="@torch.jit.script",
        jit_export_decorator="@torch.jit.export",
    )
    if is_scriptable is False:
        jit_decorator_str_map = {
            key: "" for key, value in jit_decorator_str_map.items()
        }

    remote_forward_template = CodeTemplate.from_file(
        os.path.join(TEMPLATE_DIR_PATH, "remote_module.py.template")
    )
    env = dict(
        generated_module_name=generated_module_name,
        module_interface_cls_module_name=module_interface_cls.__module__,
        module_interface_cls_name=module_interface_cls.__name__,
        arg_types=arg_types_str,
        arrow_and_return_type=arrow_and_return_type_str,
        arrow_and_future_return_type=arrow_and_future_return_type_str,
        args=args_str,
        kwargs=kwargs_str,
        **jit_decorator_str_map,
    )
    generated_code_text = remote_forward_template.substitute(env)
    out_path = os.path.join(
        TEMPLATE_DIR_PATH, "instantiated", f"{generated_module_name}.py"
    )
    write(out_path, generated_code_text)

    # From importlib doc,
    # > If you are dynamically importing a module that was created since
    # the interpreter began execution (e.g., created a Python source file),
    # you may need to call invalidate_caches() in order for the new module
    # to be noticed by the import system.
    importlib.invalidate_caches()
    generated_module = importlib.import_module(
        f"torch.distributed.nn.jit.templates.instantiated.{generated_module_name}"
    )
    return generated_module
