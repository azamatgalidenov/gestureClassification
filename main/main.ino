#include <WiFi.h>
#include <esp_camera.h>
#include <esp_http_server.h>
#include <img_converters.h>
#include <mbedtls/base64.h>
#include <TensorFlowLite_ESP32.h>

// TensorFlow Lite Micro dependencies
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

// The quantized neural network model converted to a C header file
#include "modelquantized.h" 

// Network credentials
const char* ssid = "The Procrastination station"; 
const char* pass = "Green42$";

// Memory allocation for the AI model
const int kArenaSize = 512 * 1024; 

// Class labels must match the training dataset order exactly
const char* CLASSES[] = { "Paper", "Rock", "Scissors" };

// TFLite global pointers
uint8_t* tensor_arena = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Minified HTML/JS for the client interface
// The JavaScript periodically fetches '/capture' and updates the image/prediction source.
const char index_html[] PROGMEM = R"rawliteral(<!DOCTYPE HTML><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{text-align:center;font-family:Arial;background:#222;color:#fff}img{width:100%;max-width:300px;border:2px solid #555}#pred{font-size:40px;color:#0f0;margin:10px}#scores{font-size:14px;color:#aaa;font-family:monospace}</style></head><body><h2>ESP32 Neural Cam</h2><img id="photo" src=""><div id="pred">WAITING...</div><div id="scores">Connecting...</div><script>const i=document.getElementById("photo"),p=document.getElementById("pred"),s=document.getElementById("scores");async function u(){try{const r=await fetch('/capture');if(!r.ok)throw 0;const d=await r.json();i.src="data:image/jpeg;base64,"+d.image;p.innerHTML=d.prediction;p.style.color=d.prediction==="???"?"#ff0":"#0f0";s.innerHTML=d.scores}catch(e){s.innerHTML="Retry..."}setTimeout(u,50)}window.onload=u;</script></body></html>)rawliteral";

// Helper: Converts raw model output (logits) into probabilities (0.0 - 1.0)
void softmax(float* data, int len) {
  float m = -1e38, sum = 0;
  // Find max value to prevent overflow during exponentiation
  for(int i=0; i<len; i++) if(data[i]>m) m=data[i];
  // Exponentiate and calculate sum
  for(int i=0; i<len; i++) sum += (data[i] = exp(data[i]-m));
  // Normalize
  for(int i=0; i<len; i++) data[i] /= sum;
}

// HTTP Handler: Captures image, runs inference, and returns JSON response
static esp_err_t capture_handler(httpd_req_t *req) {
  
  // 1. Acquire frame
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) return httpd_resp_send_500(req);

  // 2. Pre-process image for the AI
  // Normalize pixel data from 0-255 (int) to 0.0-1.0 (float)
  // Input size is fixed at 96x96 (9216 pixels) based on model training
  for (int i = 0; i < 9216; i++) {
      input->data.f[i] = fb->buf[i] * 0.003921569f; // equivalent to / 255.0
  }
  
  String pred = "???"; 
  String scores = "";

  // 3. Run Inference
  if (interpreter->Invoke() == kTfLiteOk) {
    
    // Process output using Softmax to get percentages
    softmax(output->data.f, 3); 
    
    float max_s = 0; 
    int max_i = -1;

    for (int i = 0; i < 3; i++) {
      float s = output->data.f[i];
      scores += String(CLASSES[i]) + ":" + String((int)(s*100)) + "% "; 
      
      if (s > max_s) { max_s = s; max_i = i; }
    }
    
    // Only return a valid prediction if confidence > 85%
    if (max_s > 0.85) pred = CLASSES[max_i];
  }

  // 4. Image Compression for Browser
  // Convert raw Grayscale buffer to JPEG for display
  uint8_t *jpg_buf = NULL; 
  size_t jpg_len = 0;
  frame2jpg(fb, 80, &jpg_buf, &jpg_len);
  
  // Release camera framebuffer immediately to allow next capture
  esp_camera_fb_return(fb);

  // 5. Encode to Base64
  // Browsers require Base64 strings to display images inside JSON
  size_t b64_len = ((jpg_len + 2) / 3) * 4 + 1;
  unsigned char* b64_buf = (unsigned char*)malloc(b64_len);
  size_t out_len = 0;
  
  mbedtls_base64_encode(b64_buf, b64_len, &out_len, jpg_buf, jpg_len);
  b64_buf[out_len] = 0; 
  free(jpg_buf); // Free raw JPEG buffer

  // 6. Send Response
  // Send JSON in chunks to avoid allocating a massive string buffer
  httpd_resp_set_type(req, "application/json");
  
  char chunk[128];
  snprintf(chunk, sizeof(chunk), "{ \"prediction\": \"%s\", \"scores\": \"%s\", \"image\": \"", pred.c_str(), scores.c_str());
  httpd_resp_send_chunk(req, chunk, strlen(chunk));
  httpd_resp_send_chunk(req, (char*)b64_buf, out_len); // Image data
  httpd_resp_send_chunk(req, "\" }", 3); // JSON closer
  httpd_resp_send_chunk(req, NULL, 0); // End of stream

  free(b64_buf);
  return ESP_OK;
}

void setup() {
  // High CPU freq required for reasonable inference time
  setCpuFrequencyMhz(240);
  Serial.begin(115200);

  // Camera Configuration (AI Thinker ESP32-CAM Pinout)
  camera_config_t config = {
    .pin_pwdn = 32, .pin_reset = -1, .pin_xclk = 0, .pin_sscb_sda = 26, .pin_sscb_scl = 27,
    .pin_d7 = 35, .pin_d6 = 34, .pin_d5 = 39, .pin_d4 = 36, .pin_d3 = 21, .pin_d2 = 19, .pin_d1 = 18, .pin_d0 = 5,
    .pin_vsync = 25, .pin_href = 23, .pin_pclk = 22,
    .xclk_freq_hz = 20000000, 
    .ledc_timer = LEDC_TIMER_0, .ledc_channel = LEDC_CHANNEL_0,
    .pixel_format = PIXFORMAT_GRAYSCALE, 
    .frame_size = FRAMESIZE_96X96,       
    .jpeg_quality = 12, 
    .fb_count = 1
  };
  esp_camera_init(&config);

  WiFi.begin(ssid, pass);
  while (WiFi.status() != WL_CONNECTED) delay(500);
  Serial.print("Camera Ready! Go to: http://"); Serial.println(WiFi.localIP());

  // TFLite Initialization
  // Allocate tensor arena in PSRAM (SPIRAM) to save internal heap
  tensor_arena = (uint8_t*)heap_caps_malloc(kArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  
  static tflite::MicroErrorReporter err_rep;
  static tflite::AllOpsResolver resolver; // Load all neural network operators
  
  static tflite::MicroInterpreter static_interpreter(
      tflite::GetModel(g_model), 
      resolver, 
      tensor_arena, 
      kArenaSize, 
      &err_rep
  );
  interpreter = &static_interpreter;

  interpreter->AllocateTensors();
  
  // Cache pointers to input/output layers
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Web Server Setup
  httpd_handle_t server = NULL;
  httpd_config_t hconf = HTTPD_DEFAULT_CONFIG();
  httpd_start(&server, &hconf);
  
  // Route: / (Index)
  httpd_uri_t uri_idx = { "/", HTTP_GET, [](httpd_req_t *r){ 
      httpd_resp_send(r, index_html, HTTPD_RESP_USE_STRLEN); return ESP_OK; 
  }, NULL };
  
  // Route: /capture (AI Logic)
  httpd_uri_t uri_cap = { "/capture", HTTP_GET, capture_handler, NULL };
  
  httpd_register_uri_handler(server, &uri_idx);
  httpd_register_uri_handler(server, &uri_cap);
}

void loop() { 
  // Main loop is empty because the HTTP server runs in a separate FreeRTOS task.
  delay(10000); 
}