import 'package:flutter/material.dart';
import 'package:flutter_application/features/home/screens/home_screen.dart';

void main(List<String> args) {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: FaceMotionDetectionScreen(),
    );
  }
}
