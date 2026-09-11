import numpy as np
from onnx import TensorProto, defs, helper
from onnx.reference import ReferenceEvaluator
imin=np.iinfo(np.int64).min
node=helper.make_node('Slice',['x','starts','ends','axes','steps'],['y'])
graph=helper.make_graph([node],'g',[helper.make_tensor_value_info('x',TensorProto.FLOAT,[3]),helper.make_tensor_value_info('starts',TensorProto.INT64,[1]),helper.make_tensor_value_info('ends',TensorProto.INT64,[1]),helper.make_tensor_value_info('axes',TensorProto.INT64,[1]),helper.make_tensor_value_info('steps',TensorProto.INT64,[1])],[helper.make_tensor_value_info('y',TensorProto.FLOAT,[None])])
model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)])
feeds={'x':np.arange(3,dtype=np.float32),'starts':np.array([imin],np.int64),'ends':np.array([imin],np.int64),'axes':np.array([0],np.int64),'steps':np.array([-1],np.int64)}
print('schema_revision=',defs.get_schema('Slice',17,'').since_version)
print('schema_negative_start_clamp=[0,dim-1]')
print('reference=',ReferenceEvaluator(model).run(None,feeds)[0].tolist())