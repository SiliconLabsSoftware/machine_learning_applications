// Auto-generated macro to instanciate and initialize opcode resolver based on TFLite flatbuffers in config directory
#ifndef SL_TFLITE_MICRO_OPCODE_RESOLVER_H
#define SL_TFLITE_MICRO_OPCODE_RESOLVER_H

#define SL_TFLITE_MICRO_OPCODE_RESOLVER(opcode_resolver) \
static tflite::MicroMutableOpResolver<7> opcode_resolver; \
opcode_resolver.AddConv2D(); \
opcode_resolver.AddMaxPool2D(); \
opcode_resolver.AddMean(); \
opcode_resolver.AddFullyConnected(); \
opcode_resolver.AddSoftmax(); \
opcode_resolver.AddQuantize(); \
opcode_resolver.AddConcatenation(); \


#endif // SL_TFLITE_MICRO_OPCODE_RESOLVER_H
