import onnx
from onnx_tf.backend import prepare
onnx_model = onnx.load( "models/esp32_model.onnx" )
tf_rep = prepare( onnx_model )
tf_rep.export_graph ("models/model_tf")