# [Speech-to-Text (STT) Transcription Service 🎤](https://github.com/ErcinDedeoglu/stt)

This repository hosts the Dockerized speech-to-text transcription service, which utilizes Whisper C++ alongside Python to provide an API for audio file transcription.

<div align="center"><img src="/assets/logo.png" width="400"></div>

## Background and Motivation

In the rapidly advancing field of machine learning, access to efficient and robust tools for everyday applications is essential. Speech-to-text transcription is one of the areas that has seen significant improvements, but deploying these models quickly and efficiently remains a challenge. Whisper C++, a high-performance transcription tool, has emerged as a powerful option, yet it still requires a streamlined pathway to deployment.

This repository was created out of a necessity to bridge the gap between the development of speech-to-text models and their deployment in real-world applications. Many existing solutions require extensive setup, intricate knowledge of systems, and can be time-consuming to deploy, creating a barrier for developers, researchers, and businesses who want to integrate transcription capabilities into their services.

The Speech-to-Text Transcription Service aims to provide a fast, reliable, and easy-to-use solution for deploying Whisper C++ models. By containerizing the service with Docker, we significantly reduce the complexity of deployment and make it possible to launch a transcription service that is both scalable and accessible.

Here are some of the key motivations behind this project:

- **Speed of Deployment**: By providing a Dockerized solution, we enable rapid deployment of the transcription service, allowing users to go from zero to a fully functioning service in minutes.
- **Ease of Use**: The provided APIs and Docker setup are designed to be as simple as possible, requiring minimal configuration and allowing for easy integration into existing workflows.
- **Accessibility**: Making Whisper C++ easily deployable opens up more opportunities for developers and organizations of all sizes to utilize state-of-the-art transcription technology.
- **Continuous Integration and Delivery**: With GitHub Actions, updates and improvements are integrated seamlessly, ensuring that the service remains up-to-date with the latest advancements from the Whisper C++ repository.

In contributing to this repository, I hope to empower individuals and organizations to harness the capabilities of Whisper C++ without the overhead of complex deployment processes, thus fostering innovation and development in the field of speech recognition.

---

## Getting Started

### Using Docker Image

For quick deployment, use the Docker images provided in the Docker registry.

For the latest stable version:
```bash
docker pull dublok/stt:latest
docker run -p 5000:5000 dublok/stt:latest
```

For the nightly build (unstable but with early access to new features):
```bash
docker pull dublok/stt:main
docker run -p 5000:5000 dublok/stt:main
```

The service should now be accessible at `http://localhost:5000`.

### Building from Source

1. Clone the repository:
```bash
git clone https://github.com/ErcinDedeoglu/stt
```

2. Build the Docker image:
```bash
docker build -t stt-service .
```

3. Run the container:
```bash
docker run -p 5000:5000 stt-service
```

---

## API Usage

To transcribe audio, make a POST request to the `/transcribe` endpoint with the audio file:

```bash
curl -X POST -F 'file=@/path/to/your/audio.wav' http://localhost:5000/transcribe
```

Make sure your audio file is in WAV format with a sample rate of 16kHz.

### Example Response

Upon successful transcription, the service will return a JSON response containing the transcription along with the timestamps for each transcribed segment. An example response might look like this:

```json
{
  "transcription": [
    {
      "start_time": "00:00:00.000",
      "end_time": "00:00:03.000",
      "text": "Welcome to our speech to text service."
    },
    {
      "start_time": "00:00:03.500",
      "end_time": "00:00:05.000",
      "text": "This is a sample transcription."
    }
  ]
}
```

If there is an error in transcription, the service might return an error response like:

```json
{
  "error": "Error in transcription"
}
```

Make sure to handle both success and error responses appropriately in your application.

--- 

Adjust the example response to match the actual output format of your transcription service. The error message should also reflect what your service would actually return in case of a failure.

## Development

### Prerequisites

- Docker
- Python 3.8
- C++ build tools (cmake, make, g++)
- ffmpeg

### Setup and Build

The `Dockerfile` in this repository details the steps to set up the environment and install dependencies necessary for running the transcription service.

### Contributing

Contributions are welcome! If you wish to contribute, please create a pull request with your proposed changes or fixes.

## Continuous Integration

This project uses GitHub Actions for continuous integration, which automates the following:

- `sync-whisper.yml`: Synchronizes with the latest tag or commit of `whisper.cpp`.
- `publish-docker.yml`: Automatically builds and pushes Docker images to the registry upon changes.

## License

This Speech-to-Text Transcription Service is made available under the [CC0 1.0 Universal](LICENSE) public domain dedication.

