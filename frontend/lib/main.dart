import 'dart:io';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:http/http.dart' as http;
import 'package:mime/mime.dart';
import 'package:http_parser/http_parser.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Image Classifier',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      darkTheme: ThemeData(
        brightness: Brightness.dark,
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      // Automatically switch theme mode based on system settings
      themeMode: ThemeMode.system,
      home: const ImageClassifierPage(),
    );
  }
}

class ImageClassifierPage extends StatefulWidget {
  const ImageClassifierPage({super.key});

  @override
  ImageClassifierPageState createState() => ImageClassifierPageState();
}

class ImageClassifierPageState extends State<ImageClassifierPage> {
  File? _selectedImage;
  String _predictedLabel = '';
  double _confidence = 0.0;
  bool _isLoading = false;

  Future<void> _pickImage() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.image,
    );

    if (result != null) {
      setState(() {
        _selectedImage = File(result.files.single.path!);
        _predictedLabel = '';
        _confidence = 0.0;
      });
      _uploadImage(_selectedImage!);
    }
  }

  Future<void> _uploadImage(File image) async {
    setState(() {
      _isLoading = true;
    });

    try {
      var request = http.MultipartRequest(
          'POST', Uri.parse('http://localhost:8000/predict'));

      // Get the MIME type of the file
      var mimeType = lookupMimeType(image.path);

      // Set MediaType based on MIME type
      var mediaType = mimeType != null
          ? MediaType.parse(mimeType)
          : MediaType('image', 'jpeg');

      request.files.add(await http.MultipartFile.fromPath(
          'file', // Field name expected by backend
          image.path,
          contentType: mediaType // Dynamically set MediaType
          ));

      var response = await request.send();

      if (response.statusCode == 200) {
        var responseData = await response.stream.bytesToString();
        var data = jsonDecode(responseData);
        setState(() {
          _predictedLabel = data['result'];
          _confidence = data['confidence'];
          _isLoading = false;
        });
      } else {
        // error handling
        setState(() {
          _isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        _isLoading = false;
      });
      // error handling
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Image Classifier'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            ElevatedButton(
              onPressed: _pickImage,
              child: const Text('Select Image'),
            ),
            const SizedBox(height: 20),
            _isLoading
                ? const CircularProgressIndicator()
                : _selectedImage != null
                    ? Image.file(_selectedImage!, height: 300, width: 300)
                    : const SizedBox(height: 300, width: 300),
            const SizedBox(height: 20),
            Text('Predicted Label: $_predictedLabel'),
            Text('Confidence: ${_confidence.toStringAsFixed(2)}%'),
          ],
        ),
      ),
    );
  }
}
