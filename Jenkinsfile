pipeline {
    agent any

    environment {
        DOCKER_IMAGE = "fraud-detection-app"
        DOCKERHUB_REPO = "tada8102/fraud-detection-app"
        CREDENTIALS_ID = "dockerhub" // Jenkins credentials ID for Docker Hub
    }

    stages {
        stage('Login to Docker Hub') {
            steps {
                withCredentials([usernamePassword(credentialsId: "${CREDENTIALS_ID}", usernameVariable: 'DOCKERHUB_USER', passwordVariable: 'DOCKERHUB_PASS')]) {
                    sh 'echo $DOCKERHUB_PASS | docker login -u $DOCKERHUB_USER --password-stdin'
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    docker.build(DOCKER_IMAGE)
                }
            }
        }

        stage('Test Docker Image') {
            steps {
                script {
                    docker.image(DOCKER_IMAGE).inside {
                        sh 'python -m unittest discover'
                    }
                }
            }
        }

        stage('Tag & Push Docker Image') {
            steps {
                script {
                    sh "docker tag ${DOCKER_IMAGE} ${DOCKERHUB_REPO}:latest"
                    sh "docker push ${DOCKERHUB_REPO}:latest"
                }
            }
        }

        stage('Cleanup') {
            steps {
                sh "docker system prune -f"
            }
        }
    }

    post {
        always {
            sh "docker logout"
        }
        success {
            echo "Docker image pushed successfully!"
        }
        failure {
            echo "Pipeline failed. Check logs for errors."
        }
    }
}
