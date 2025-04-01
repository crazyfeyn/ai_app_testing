import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class FaceMotionDetector {
  Interpreter? _interpreter;
  bool isProcessing = false;

  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(
          'assets/models/your_face_motion_model.tflite');
      print("Model loaded successfully");
    } catch (e) {
      print("Error loading model: $e");
    }
  }

  Future<Map<String, dynamic>?> detectFaceMotion(
      CameraImage cameraImage) async {
    if (isProcessing || _interpreter == null) return null;
    isProcessing = true;

    try {
      print("Starting face motion detection");

      // Convert CameraImage to format the model expects
      final imgLib = _convertCameraImage(cameraImage);
      if (imgLib == null) {
        print("Failed to convert camera image");
        return null;
      }
      print("Image conversion successful");

      // Resize image to model input size
      final resizedImg = img.copyResize(
        imgLib,
        width: 224, // Adjust based on your model's input size
        height: 224,
      );
      print("Image resized to 224x224");

      // Convert to input tensor format (float32 buffer)
      final inputBuffer = _imageToByteBuffer(resizedImg);
      print("Converted to input tensor format");

      // Setup output buffer
      final outputBuffer = <int, Object>{};

      // Get the output shape from the model
      final outputShape = _interpreter!.getOutputTensor(0).shape;

      // Create the output tensor with the right shape
      outputBuffer[0] =
          List<double>.filled(outputShape.reduce((a, b) => a * b), 0)
              .reshape(outputShape);

      print("Running model inference...");
      // Run inference
      _interpreter!.runForMultipleInputs([inputBuffer], outputBuffer);
      print("Model inference completed");

      // Process results - adjust based on your model's output format
      final outputList = outputBuffer[0] as List;

      // Assuming your model outputs lip movement and jaw position probabilities
      final lipMovement = outputList[0] > 0.5;
      final jawPosition = outputList[1] > 0.5 ? "Correct" : "Incorrect";

      print(
          "Detection results: lipMovement=$lipMovement, jawPosition=$jawPosition");

      return {"lip_movement": lipMovement, "jaw_position": jawPosition};
    } catch (e) {
      print("Error in detectFaceMotion: $e");
      return null;
    } finally {
      isProcessing = false;
    }
  }

  img.Image? _convertCameraImage(CameraImage image) {
    try {
      // For YUV_420_888 format which is common for Android cameras
      final int width = image.width;
      final int height = image.height;

      // Create a new image
      final img.Image rgbImage = img.Image(width: width, height: height);

      // Convert YUV to RGB
      final yBuffer = image.planes[0].bytes;
      final uBuffer = image.planes[1].bytes;
      final vBuffer = image.planes[2].bytes;

      final yRowStride = image.planes[0].bytesPerRow;
      final uvRowStride = image.planes[1].bytesPerRow;
      final uvPixelStride = image.planes[1].bytesPerPixel ?? 1;

      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          final int yIndex = y * yRowStride + x;

          final int uvIndex = uvPixelStride * (x ~/ 2) + (y ~/ 2) * uvRowStride;

          // YUV to RGB conversion
          int yValue = yBuffer[yIndex] & 0xff;

          int uValue = 0;
          int vValue = 0;
          if (uvIndex < uBuffer.length) {
            uValue = uBuffer[uvIndex] & 0xff;
          }
          if (uvIndex < vBuffer.length) {
            vValue = vBuffer[uvIndex] & 0xff;
          }

          yValue = yValue - 16;
          uValue = uValue - 128;
          vValue = vValue - 128;

          int r = (1.164 * yValue + 1.596 * vValue).round();
          int g = (1.164 * yValue - 0.813 * vValue - 0.391 * uValue).round();
          int b = (1.164 * yValue + 2.018 * uValue).round();

          r = r.clamp(0, 255);
          g = g.clamp(0, 255);
          b = b.clamp(0, 255);

          // Add the alpha channel value (255 for fully opaque)
          rgbImage.setPixelRgba(x, y, r, g, b, 255);
        }
      }

      return rgbImage;
    } catch (e) {
      print("Error converting camera image: $e");
      return null;
    }
  }

  Float32List _imageToByteBuffer(img.Image image) {
    final Float32List result = Float32List(1 * 224 * 224 * 3);
    var buffer = Float32List.view(result.buffer);
    int pixelIndex = 0;

    for (var y = 0; y < 224; y++) {
      for (var x = 0; x < 224; x++) {
        // Access RGB channels
        int r = 0, g = 0, b = 0;

        try {
          final pixel = image.getPixel(x, y);
          // Cast the num values to int
          r = pixel.r.toInt();
          g = pixel.g.toInt();
          b = pixel.b.toInt();
        } catch (e) {
          print("Warning: Failed to get pixel color. Using default values.");
          r = g = b = 128; // Default color (mid-gray)
        }

        // Normalize to [-1, 1] range
        buffer[pixelIndex++] = r / 127.5 - 1.0;
        buffer[pixelIndex++] = g / 127.5 - 1.0;
        buffer[pixelIndex++] = b / 127.5 - 1.0;
      }
    }

    return result;
  }

  void dispose() {
    _interpreter?.close();
  }
}
