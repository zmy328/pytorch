#include <torch/csrc/python_headers.h>

#include <torch/csrc/distributed/rpc/process_group_agent.h>
#include <torch/csrc/distributed/rpc/py_rref.h>
#include <torch/csrc/distributed/rpc/python_functions.h>
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/torchscript_functions.h>
#include <torch/csrc/distributed/rpc/types.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_compat.h>
#include <torch/types.h>

#include <pybind11/chrono.h>
#include <pybind11/operators.h>

namespace torch {
namespace distributed {
namespace rpc {

namespace {

constexpr std::chrono::milliseconds kDeleteAllUsersTimeout(100000);

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

PyObject* rpc_init(PyObject* /* unused */) {
  auto rpc_module =
      THPObjectPtr(PyImport_ImportModule("torch.distributed.rpc"));
  if (!rpc_module) {
    throw python_error();
  }

  auto module = py::handle(rpc_module).cast<py::module>();

  auto rpcBackendOptions =
      shared_ptr_class_<RpcBackendOptions>(
          module,
          "RpcBackendOptions",
          R"(An abstract structure encapsulating the options passed into the RPC
            backend. An instance of this class can be passed in to
            :meth:`~torch.distributed.rpc.init_rpc` in order to initialize RPC
            with specific configurations, such as the RPC timeout and
            ``init_method`` to be used. )")
          .def_readwrite(
              "rpc_timeout",
              &RpcBackendOptions::rpcTimeoutSeconds,
              R"(A float indicating the timeout to use for all
                RPCs. If an RPC does not complete in this timeframe, it will
                complete with an exception indicating that it has timed out.)")
          .def_readwrite(
              "init_method",
              &RpcBackendOptions::initMethod,
              R"(URL specifying how to initialize the process group.
                Default is ``env://``)");

  // The following C++ constants need to be cast so they can be used from
  // python.
  module.attr("_DEFAULT_RPC_TIMEOUT_SEC") = py::cast(kDefaultRpcTimeoutSeconds);
  module.attr("_UNSET_RPC_TIMEOUT") = py::cast(kUnsetRpcTimeout);
  module.attr("_DEFAULT_INIT_METHOD") = py::cast(kDefaultInitMethod);

  auto workerInfo =
      shared_ptr_class_<WorkerInfo>(
          module,
          "WorkerInfo",
          R"(A structure that encapsulates information of a worker in the system.
            Contains the name and ID of the worker. This class is not meant to
            be constructed directly, rather, an instance can be retrieved
            through :meth:`~torch.distributed.rpc.get_worker_info` and the
            result can be passed in to functions such as
            :meth:`~torch.distributed.rpc.rpc_sync`, :class:`~torch.distributed.rpc.rpc_async`,
            :meth:`~torch.distributed.rpc.remote` to avoid copying a string on
            every invocation.)")
          .def(
              py::init<std::string, worker_id_t>(),
              py::arg("name"),
              py::arg("id"))
          .def_readonly(
              "name", &WorkerInfo::name_, R"(The name of the worker.)")
          .def_readonly(
              "id",
              &WorkerInfo::id_,
              R"(Globally unique id to identify the worker.)")
          .def("__eq__", &WorkerInfo::operator==, py::is_operator())
          // pybind11 suggests the syntax  .def(hash(py::self)), with the
          // unqualified "hash" function call. However the
          // argument-dependent lookup for the function "hash" doesn't get
          // triggered in this context because it conflicts with the struct
          // torch::hash, so  we need to use the qualified name
          // py::detail::hash, which unfortunately is in a detail namespace.
          .def(py::detail::hash(py::self));

  auto rpcAgent =
      shared_ptr_class_<RpcAgent>(module, "RpcAgent")
          .def(
              "join", &RpcAgent::join, py::call_guard<py::gil_scoped_release>())
          .def(
              "sync", &RpcAgent::sync, py::call_guard<py::gil_scoped_release>())
          .def(
              "get_worker_infos",
              &RpcAgent::getWorkerInfos,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "get_debug_info",
              &RpcAgent::getDebugInfo,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "get_metrics",
              &RpcAgent::getMetrics,
              py::call_guard<py::gil_scoped_release>());

  auto pyRRef =
      shared_ptr_class_<PyRRef>(module, "RRef", R"(
          A class encapsulating a reference to a value of some type on a remote
          worker. This handle will keep the referenced remote value alive on the
          worker. A ``UserRRef`` will be deleted when 1) no references to it in
          both the application code and in the local RRef context, or 2) the
          application has called a graceful shutdown. Invoking methods on a
          deleted RRef leads to undefined behaviors. RRef implementation only
          offers best-effort error detection, and applications should not use
          ``UserRRefs`` after ``rpc.shutdown()``.

          .. warning::
              RRefs can only be serialized and deserialized by the RPC module.
              Serializing and deserializing RRefs without RPC (e.g., Python
              pickle, torch :meth:`~torch.save` / :meth:`~torch.load`,
              JIT :meth:`~torch.jit.save` / :meth:`~torch.jit.load`, etc.) will
              lead to errors.

          Example::
              Following examples skip RPC initialization and shutdown code
              for simplicity. Refer to RPC docs for those details.

              1. Create an RRef using rpc.remote

              >>> import torch
              >>> import torch.distributed.rpc as rpc
              >>> rref = rpc.remote("worker1", torch.add, args=(torch.ones(2), 3))
              >>> # get a copy of value from the RRef
              >>> x = rref.to_here()

              2. Create an RRef from a local object

              >>> import torch
              >>> from torch.distributed.rpc import RRef
              >>> x = torch.zeros(2, 2)
              >>> rref = RRef(x)

              3. Share an RRef with other workers

              >>> # On both worker0 and worker1:
              >>> def f(rref):
              >>>   return rref.to_here() + 1

              >>> # On worker0:
              >>> import torch
              >>> import torch.distributed.rpc as rpc
              >>> from torch.distributed.rpc import RRef
              >>> rref = RRef(torch.zeros(2, 2))
              >>> # the following RPC shares the rref with worker1, reference
              >>> # count is automatically updated.
              >>> rpc.rpc_sync("worker1", f, args(rref,))
          )")
          .def(
              py::init<const py::object&, const py::object&>(),
              py::arg("value"),
              py::arg("type_hint") = py::none())
          .def(
              // not releasing GIL here to avoid context switch on getters
              "is_owner",
              &PyRRef::isOwner,
              R"(
                  Returns whether or not the current node is the owner of this
                  ``RRef``.
              )")
          .def(
              "confirmed_by_owner",
              &PyRRef::confirmedByOwner,
              R"(
                  Returns whether this ``RRef`` has been confirmed by the owner.
                  ``OwnerRRef`` always returns true, while ``UserRRef`` only
                  returns true when the owner knowns about this ``UserRRef``.
              )")
          .def(
              // not releasing GIL here to avoid context switch on getters
              "owner",
              &PyRRef::owner,
              R"(
                  Returns worker information of the node that owns this ``RRef``.
              )")
          .def(
              // not releasing GIL here to avoid context switch on getters
              "owner_name",
              &PyRRef::ownerName,
              R"(
                  Returns worker name of the node that owns this ``RRef``.
              )")
          .def(
              "to_here",
              &PyRRef::toHere,
              py::call_guard<py::gil_scoped_release>(),
              R"(
                  Blocking call that copies the value of the RRef from the owner
                  to the local node and returns it. If the current node is the
                  owner, returns a reference to the local value.
              )")
          .def(
              "local_value",
              &PyRRef::localValue,
              py::call_guard<py::gil_scoped_release>(),
              R"(
                  If the current node is the owner, returns a reference to the
                  local value. Otherwise, throws an exception.
              )")
          .def(
              "rpc_sync",
              [&](PyRRef& self) {
                return self.createRRefProxy(self, RRefProxyType::RPC_SYNC);
              },
              py::call_guard<py::gil_scoped_release>(),
              R"(
                  Create a helper proxy to easily launch an ``rpc_sync`` using
                  the owner of the RRef as the destination to run functions on
                  the object referenced by this RRef. More specifically,
                  ``rref.rpc_sync().func_name(*args, **kwargs)`` is the same as
                  the following:

                  >>> def run(rref, func_name, args, kwargs):
                  >>>   return getattr(rref.local_value(), func_name)(*args, **kwargs)
                  >>>
                  >>> rpc.rpc_sync(rref.owner(), run, args=(rref, func_name, args, kwargs))

                  Example::
                      >>> from torch.distributed import rpc
                      >>> rref = rpc.remote("worker1", torch.add, args=(torch.zeros(2, 2), 1))
                      >>> rref.rpc_sync().size()  # returns torch.Size([2, 2])
                      >>> rref.rpc_sync().view(1, 4)  # returns tensor([[1., 1., 1., 1.]])
              )")
          .def(
              "rpc_async",
              [&](PyRRef& self) {
                return self.createRRefProxy(self, RRefProxyType::RPC_ASYNC);
              },
              py::call_guard<py::gil_scoped_release>(),
              R"(
                  Create a helper proxy to easily launch an ``rpc_async`` using
                  the owner of the RRef as the destination to run functions on
                  the object referenced by this RRef. More specifically, 
                  ``rref.rpc_async().func_name(*args, **kwargs)`` is the same as
                  the following:

                  >>> def run(rref, func_name, args, kwargs):
                  >>>   return getattr(rref.local_value(), func_name)(*args, **kwargs)
                  >>>
                  >>> rpc.rpc_async(rref.owner(), run, args=(rref, func_name, args, kwargs))

                  Example::
                      >>> from torch.distributed import rpc
                      >>> rref = rpc.remote("worker1", torch.add, args=(torch.zeros(2, 2), 1))
                      >>> rref.rpc_async().size().wait()  # returns torch.Size([2, 2])
                      >>> rref.rpc_async().view(1, 4).wait()  # returns tensor([[1., 1., 1., 1.]])
              )")
          .def(
              "remote",
              [&](PyRRef& self) {
                return self.createRRefProxy(self, RRefProxyType::REMOTE);
              },
              py::call_guard<py::gil_scoped_release>(),
              R"(
                  Create a helper proxy to easily launch an ``remote`` using
                  the owner of the RRef as the destination to run functions on
                  the object referenced by this RRef. More specifically, 
                  ``rref.remote().func_name(*args, **kwargs)`` is the same as
                  the following:

                  >>> def run(rref, func_name, args, kwargs):
                  >>>   return getattr(rref.local_value(), func_name)(*args, **kwargs)
                  >>>
                  >>> rpc.remote(rref.owner(), run, args=(rref, func_name, args, kwargs))

                  Example::
                      >>> from torch.distributed import rpc
                      >>> rref = rpc.remote("worker1", torch.add, args=(torch.zeros(2, 2), 1))
                      >>> rref.remote().size().to_here()  # returns torch.Size([2, 2])
                      >>> rref.remote().view(1, 4).to_here()  # returns tensor([[1., 1., 1., 1.]])
              )")
          .def(
              py::pickle(
                  [](const PyRRef& self) {
                    TORCH_CHECK(
                        false,
                        "Can not pickle rref in python pickler, rref can only be pickled when using RPC");
                    // __getstate__
                    return self.pickle();
                  },
                  [](py::tuple t) { // NOLINT
                    TORCH_CHECK(
                        false,
                        "Can not unpickle rref in python pickler, rref can only be unpickled when using RPC");
                    // __setstate__
                    return PyRRef::unpickle(t);
                  }),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_serialize",
              &PyRRef::pickle,
              py::call_guard<py::gil_scoped_release>())
          .def_static(
              "_deserialize",
              &PyRRef::unpickle,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_get_future",
              &PyRRef::getFuture,
              py::call_guard<py::gil_scoped_release>(),
              R"(
                  Returns the future that corresponds to the creation of this RRef
                  on the remote node. This is for internal use cases such as profiling
                  only.
              )")
          // not releasing GIL to avoid context switch
          .def("__str__", &PyRRef::str);

  // future.wait() should not be called after shutdown(), e.g.,
  // pythonRpcHandler is cleaned up in shutdown(), after
  // shutdown(), python objects returned from rpc python call can not be
  // resolved.
  // TODO Once python object can be tagged as IValue and c10::ivalue::Future is
  // implemented as generic Future<IValue>, we can consider all rpc call
  // to return a future<IValue> later on.
  auto future = shared_ptr_class_<FutureMessage>(module, "Future")
                    .def(
                        "wait",
                        [&](FutureMessage& fut) { return toPyObj(fut.wait()); },
                        py::call_guard<py::gil_scoped_release>(),
                        R"(
Wait on future to complete and return the object it completed with.
If the future completes with an error, an exception is thrown.
              )");

  shared_ptr_class_<ProcessGroupRpcBackendOptions>(
      module,
      "ProcessGroupRpcBackendOptions",
      rpcBackendOptions,
      R"(
          The backend options class for ``ProcessGroupAgent``, which is derived
          from ``RpcBackendOptions``.

          Arguments:
              num_send_recv_threads (int, optional): The number of threads in
                  the thread-pool used by ``ProcessGroupAgent`` (default: 4).
              rpc_timeout (float, optional): The default timeout, in seconds,
                  for RPC requests (default: 60 seconds). If the
                  RPC has not completed in this timeframe, an exception
                  indicating so will be raised. Callers can override this
                  timeout for individual RPCs in
                  :meth:`~torch.distributed.rpc.rpc_sync` and
                  :meth:`~torch.distributed.rpc.rpc_async` if necessary.
              init_method (str, optional): The URL to initialize
                  ``ProcessGroupGloo`` (default: ``env://``).


          Example::
              >>> import datetime, os
              >>> from torch.distributed import rpc
              >>> os.environ['MASTER_ADDR'] = 'localhost'
              >>> os.environ['MASTER_PORT'] = '29500'
              >>>
              >>> rpc.init_rpc(
              >>>     "worker1",
              >>>     rank=0,
              >>>     world_size=2,
              >>>     rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(
              >>>         num_send_recv_threads=16,
              >>>         20 # 20 second timeout
              >>>     )
              >>> )
              >>>
              >>> # omitting init_rpc invocation on worker2
      )")
      .def(
          py::init<int, float, std::string>(),
          py::arg("num_send_recv_threads") = kDefaultNumSendRecvThreads,
          py::arg("rpc_timeout") = kDefaultRpcTimeoutSeconds,
          py::arg("init_method") = kDefaultInitMethod)
      .def_readwrite(
          "num_send_recv_threads",
          &ProcessGroupRpcBackendOptions::numSendRecvThreads,
          R"(
              The number of threads in the thread-pool used by ProcessGroupAgent.
          )");

  module.attr("_DEFAULT_NUM_SEND_RECV_THREADS") =
      py::cast(kDefaultNumSendRecvThreads);

  shared_ptr_class_<ProcessGroupAgent>(module, "ProcessGroupAgent", rpcAgent)
      .def(
          py::init<
              std::string,
              std::shared_ptr<::c10d::ProcessGroup>,
              int,
              std::chrono::milliseconds>(),
          py::arg("name"),
          py::arg("process_group"),
          py::arg("num_send_recv_threads"),
          py::arg("rpc_timeout"))
      .def(
          "get_worker_info",
          (const WorkerInfo& (ProcessGroupAgent::*)(void)const) &
              RpcAgent::getWorkerInfo,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_worker_info",
          (const WorkerInfo& (ProcessGroupAgent::*)(const std::string&)const) &
              ProcessGroupAgent::getWorkerInfo,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_worker_infos",
          (std::vector<WorkerInfo>(ProcessGroupAgent::*)() const) &
              ProcessGroupAgent::getWorkerInfos,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "join",
          &ProcessGroupAgent::join,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "shutdown",
          &ProcessGroupAgent::shutdown,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "sync",
          &ProcessGroupAgent::sync,
          py::call_guard<py::gil_scoped_release>());

  module.def("_is_current_rpc_agent_set", &RpcAgent::isCurrentRpcAgentSet);

  module.def("_get_current_rpc_agent", &RpcAgent::getCurrentRpcAgent);

  module.def(
      "_set_and_start_rpc_agent",
      [](const std::shared_ptr<RpcAgent>& rpcAgent) {
        RpcAgent::setCurrentRpcAgent(rpcAgent);
        // Initializing typeResolver inside RpcAgent constructor will make
        // RpcAgent have python dependency. To avoid RpcAgent to have python
        // dependency, setTypeResolver() here.
        std::shared_ptr<TypeResolver> typeResolver =
            std::make_shared<TypeResolver>([&](const c10::QualifiedName& qn) {
              auto typePtr = PythonRpcHandler::getInstance().parseTypeFromStr(
                  qn.qualifiedName());
              return c10::StrongTypePtr(
                  PythonRpcHandler::getInstance().jitCompilationUnit(),
                  std::move(typePtr));
            });
        rpcAgent->setTypeResolver(typeResolver);
        rpcAgent->start();
      },
      py::call_guard<py::gil_scoped_release>());

  module.def("_reset_current_rpc_agent", []() {
    RpcAgent::setCurrentRpcAgent(nullptr);
  });

  module.def(
      "_delete_all_user_rrefs",
      [](std::chrono::milliseconds timeoutMillis) {
        RRefContext::getInstance().delAllUsers(timeoutMillis);
      },
      py::arg("timeout") = kDeleteAllUsersTimeout);

  module.def("_destroy_rref_context", [](bool ignoreRRefLeak) {
    // NB: do not release GIL in the function. The destroyInstance() method
    // returns a list of deleted OwnerRRefs that hold py::object instances.
    // Clearing those OwnerRRefs are likely to trigger Python deref, which
    // requires GIL.
    RRefContext::getInstance().destroyInstance(ignoreRRefLeak).clear();
  });

  module.def("_rref_context_get_debug_info", []() {
    return RRefContext::getInstance().getDebugInfo();
  });

  module.def(
      "_cleanup_python_rpc_handler",
      []() { PythonRpcHandler::getInstance().cleanup(); },
      py::call_guard<py::gil_scoped_release>());

  module.def(
      "_invoke_rpc_builtin",
      [](const WorkerInfo& dst,
         const std::string& opName,
         const float rpcTimeoutSeconds,
         const py::args& args,
         const py::kwargs& kwargs) {
        DCHECK(PyGILState_Check());
        return pyRpcBuiltin(dst, opName, args, kwargs, rpcTimeoutSeconds);
      },
      py::call_guard<py::gil_scoped_acquire>());

  module.def(
      "_invoke_rpc_python_udf",
      [](const WorkerInfo& dst,
         std::string& pickledPythonUDF,
         std::vector<torch::Tensor>& tensors,
         const float rpcTimeoutSeconds) {
        DCHECK(!PyGILState_Check());
        return pyRpcPythonUdf(
            dst, pickledPythonUDF, tensors, rpcTimeoutSeconds);
      },
      py::call_guard<py::gil_scoped_release>());

  module.def(
      "_invoke_rpc_torchscript",
      [](const std::string& dstWorkerName,
         const std::string& qualifiedNameStr,
         const py::tuple& argsTuple,
         const py::dict& kwargsDict,
         const float rpcTimeoutSeconds) {
        // No need to catch exception here, if function can not be found,
        // exception will be thrown in get_function() call; if args do not match
        // with function schema, exception will be thrown in
        // createStackForSchema() call.
        DCHECK(!PyGILState_Check());
        const c10::QualifiedName qualifiedName(qualifiedNameStr);
        auto functionSchema = PythonRpcHandler::getInstance()
                                  .jitCompilationUnit()
                                  ->get_function(qualifiedName)
                                  .getSchema();
        Stack stack;
        {
          // Acquire GIL for py::args and py::kwargs processing.
          py::gil_scoped_acquire acquire;
          stack = torch::jit::createStackForSchema(
              functionSchema,
              argsTuple.cast<py::args>(),
              kwargsDict.cast<py::kwargs>(),
              c10::nullopt);
        }
        DCHECK(!PyGILState_Check());
        c10::intrusive_ptr<c10::ivalue::Future> fut = rpcTorchscript(
            dstWorkerName,
            qualifiedName,
            functionSchema,
            stack,
            rpcTimeoutSeconds);
        return torch::jit::PythonFutureWrapper(fut);
      },
      py::call_guard<py::gil_scoped_release>());

  module.def(
      "_invoke_remote_builtin",
      [](const WorkerInfo& dst,
         const std::string& opName,
         const py::args& args,
         const py::kwargs& kwargs) {
        DCHECK(PyGILState_Check());
        return pyRemoteBuiltin(dst, opName, args, kwargs);
      },
      py::call_guard<py::gil_scoped_acquire>());

  module.def(
      "_invoke_remote_torchscript",
      [](const std::string& dstWorkerName,
         const std::string& qualifiedNameStr,
         const py::args& args,
         const py::kwargs& kwargs) {
        DCHECK(!PyGILState_Check());
        auto qualifiedName = c10::QualifiedName(qualifiedNameStr);
        auto functionSchema = PythonRpcHandler::getInstance()
                                  .jitCompilationUnit()
                                  ->get_function(qualifiedName)
                                  .getSchema();
        Stack stack;
        // Acquire GIL for py::args and py::kwargs processing.
        {
          pybind11::gil_scoped_acquire ag;
          stack = torch::jit::createStackForSchema(
              functionSchema, args, kwargs, c10::nullopt);
        }
        DCHECK(!PyGILState_Check());
        auto rrefPtr = remoteTorchscript(
            dstWorkerName, qualifiedName, functionSchema, stack);
        return PyRRef(rrefPtr);
      },
      py::call_guard<py::gil_scoped_release>());

  module.def(
      "_invoke_remote_python_udf",
      [](const WorkerInfo& dst,
         std::string& pickledPythonUDF,
         std::vector<torch::Tensor>& tensors) {
        DCHECK(!PyGILState_Check());
        return pyRemotePythonUdf(dst, pickledPythonUDF, tensors);
      },
      py::call_guard<py::gil_scoped_release>(),
      py::arg("dst"),
      py::arg("pickledPythonUDF"),
      py::arg("tensors"));

  module.def(
      "get_rpc_timeout",
      []() {
        const float msToSecDiv = 1000.f;
        return RpcAgent::getCurrentRpcAgent()->getRpcTimeout().count() /
            msToSecDiv;
      },
      R"(
          Retrieve the default timeout for all RPCs that was set during RPC initialization.
          The returned value will be in seconds.
          Returns:
            ``float`` indicating the RPC timeout in seconds.
      )");

  module.def(
      "enable_gil_profiling",
      [](bool flag) {
        RpcAgent::getCurrentRpcAgent()->enableGILProfiling(flag);
      },
      R"(
    Set whether GIL wait times should be enabled or not. This incurs a slight
    overhead cost. Default is disabled for performance reasons.

    Arguments:
        flag (bool): True to set GIL profiling, False to disable.
      )");

  module.def(
      "_set_rpc_timeout",
      [](const float rpcTimeoutSeconds) {
        const auto secToMsConversion = 1000;
        auto rpcTimeout = std::chrono::milliseconds(
            static_cast<int>(rpcTimeoutSeconds * secToMsConversion));
        RpcAgent::getCurrentRpcAgent()->setRpcTimeout(rpcTimeout);
      },
      R"(
          Set the default timeout for all RPCs. The input unit is expected to be
          in seconds.If an RPC is not completed within this time, an exception
          indicating it has timed out will be raised. To control timeout for
          specific RPCs, a timeout parameter can be passed into
          :meth:`~torch.distributed.rpc.rpc_sync` and
          :meth:`~torch.distributed.rpc.rpc_async`.

          Arguments:
            rpcTimeoutSeconds (float): Timeout value in seconds.
      )");

  Py_RETURN_TRUE;
}

} // namespace

static PyMethodDef methods[] = { // NOLINT
    {"_rpc_init", (PyCFunction)rpc_init, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef* python_functions() {
  return methods;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
