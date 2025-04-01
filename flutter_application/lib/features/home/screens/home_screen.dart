import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_application/features/service/face_motion_detector.dart';

class FaceMotionDetectionScreen extends StatefulWidget {
  @override
  _FaceMotionDetectionScreenState createState() =>
      _FaceMotionDetectionScreenState();
}

class _FaceMotionDetectionScreenState extends State<FaceMotionDetectionScreen> {
  late List<CameraDescription> cameras;
  CameraController? cameraController;
  FaceMotionDetector detector = FaceMotionDetector();
  Map<String, dynamic>? motionAnalysis;
  bool isDetecting = false;

  @override
  void initState() {
    super.initState();
    initializeCamera();
    detector.loadModel();
  }

  Future<void> initializeCamera() async {
    try {
      cameras = await availableCameras();
      if (cameras.isNotEmpty) {
        // Use front camera if available
        final frontCamera = cameras.firstWhere(
          (camera) => camera.lensDirection == CameraLensDirection.front,
          orElse: () => cameras.first,
        );

        cameraController = CameraController(
          frontCamera,
          ResolutionPreset.medium,
          enableAudio: false,
          imageFormatGroup: ImageFormatGroup
              .yuv420, // Specify format for better compatibility
        );

        await cameraController!.initialize();
        if (mounted) {
          setState(() {});
          startDetection();
        }
      } else {
        print("No cameras available on device");
      }
    } catch (e) {
      print("Error initializing camera: $e");
    }
  }

  void startDetection() {
    if (cameraController != null && cameraController!.value.isInitialized) {
      cameraController!.startImageStream((image) async {
        if (!isDetecting) {
          isDetecting = true;
          try {
            final results = await detector.detectFaceMotion(image);
            if (mounted && results != null) {
              setState(() {
                motionAnalysis = results;
              });
            }
          } catch (e) {
            print("Error in face detection: $e");
          } finally {
            isDetecting = false;
          }
        }
      });
    }
  }

  @override
  void dispose() {
    cameraController?.dispose();
    detector.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Child Face Motion Detection')),
      body: cameraController == null || !cameraController!.value.isInitialized
          ? Center(child: CircularProgressIndicator())
          : Stack(
              children: [
                CameraPreview(cameraController!),
                if (motionAnalysis != null)
                  Positioned(
                    bottom: 0,
                    left: 0,
                    right: 0,
                    child: Container(
                      padding: EdgeInsets.all(8),
                      color: Colors.black54,
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            "Lip Movement: ${motionAnalysis!["lip_movement"] ? "Correct" : "Incorrect"}",
                            style: TextStyle(color: Colors.white, fontSize: 16),
                          ),
                          Text(
                            "Jaw Position: ${motionAnalysis!["jaw_position"]}",
                            style: TextStyle(color: Colors.white, fontSize: 16),
                          ),
                        ],
                      ),
                    ),
                  ),
              ],
            ),
    );
  }
}
