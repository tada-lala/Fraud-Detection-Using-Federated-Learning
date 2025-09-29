pipeline {
    agent any

    environment {
        DOCKER_IMAGE = "fraud-detection-app"
    }

    stages {
        stage('Build') {
            steps {
                script {
                    docker.build(DOCKER_IMAGE)
                }
            }
        }
        stage('Test') {
            steps {
                script {
                    docker.image(DOCKER_IMAGE).inside {
                        sh 'python -m unittest discover'
                    }
                }
            }
        }
    }
}
