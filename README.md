# SleepDetection

## Requirements

- Android Studio 4.0 (installed on a Linux, Mac or Windows machine)

- Android device with Minimum SDK version == 21 in
  [developer mode](https://developer.android.com/studio/debug/dev-options)
  with USB debugging enabled

- USB cable (to connect Android device to your computer)

## Build and run

### Step 1. Clone the source code

Clone the Sleep Detection GitHub repository to your computer to get the application.

```bash
git clone https://github.com/Ahwar/SleepDetection.git
```

To open the source code in Android Studio, open Android
Studio and select `Open an existing project`, select the folder
where you cloned source code. for example: `/home/Downloads/SleepDetection/`

<img src="images/classifydemo_img1.png?raw=true" />

### Step 2. Build the Android Studio project

Select `Build -> Make Project` and check that the project builds successfully.
You will need Android SDK configured in the settings. You'll need at least SDK version 21. The `build.gradle` file will prompt you to download any missing
libraries.

<img src="images/classifydemo_img4.png?raw=true" style="width: 60%" />

<img src="images/classifydemo_img2.png?raw=true" style="width: 60%" />

<aside class="note"><b>Note:</b><p>`build.gradle` is configured to use
TensorFlow Lite's build.</p><p>If you see a build error related to
compatibility with Tensorflow Lite's Java API (for example, `method X is
undefined for type Interpreter`), there has likely been a backwards compatible
change to the API.

### Step 3. Install and run the app

Connect the Android device to the computer and be sure to approve any ADB
permission prompts that appear on your phone. Select `Run -> Run app.` Select
the deployment target in the connected devices to the device on which the app
will be installed. This will install the app on the device.

<img src="images/classifydemo_img5.png?raw=true" style="width: 70%" />

<img src="images/classifydemo_img6.png?raw=true" style="width: 70%" />
   
<img src="images/classifydemo_img7.png?raw=true" style="width: 60%" />
<img src="images/classifydemo_img8.png?raw=true" style="width: 70%" />

To test the app, open the app called `Sleep Detection` on your device. When you run
the app the first time, the app will request permission to access the camera.
