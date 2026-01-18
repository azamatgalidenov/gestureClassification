import tensorflow as tf

# 1. CONVERT SAVEDMODEL TO TFLITE
converter = tf.lite.TFLiteConverter.from_saved_model("trained_models/model_tf")

# Perform the conversion (Float32)
tflite_model = converter.convert()

# Calculate size for C++ variable
model_len = len(tflite_model)
print(f"Model Size: {model_len / 1024:.2f} KB")

# 2. GENERATE C HEADER FILE (.h)
with open("trained_models/modelnooptimized.h", "w", encoding="utf-8") as f:
    
    # <pgmspace.h>: Required for PROGMEM (storing data in Flash instead of RAM).
    f.write("#include <pgmspace.h>\n\n")
    
    # g_model_len: The interpreter needs to know exactly how many bytes to read.
    f.write(f"const int g_model_len = {model_len};\n")
    
    f.write("const unsigned char g_model[] __attribute__((aligned(16))) PROGMEM = {\n")

    # Convert binary bytes to hex strings (e.g., 255 -> "0xff")
    hex_values = [f"0x{b:02x}" for b in tflite_model]

    # Write in chunks of 12 bytes per line for code readability
    # This loop replaces the complex buffer logic in the previous code.
    for i in range(0, len(hex_values), 12):
        chunk = hex_values[i:i+12]
        
        # Logic: Add a comma after the line unless it is the absolute last line.
        is_last_chunk = (i + 12 >= len(hex_values))
        line_end = "\n" if is_last_chunk else ",\n"
        
        f.write("  " + ", ".join(chunk) + line_end)

    f.write("};\n")

print("Success: trained_models/modelnooptimized.h created.")