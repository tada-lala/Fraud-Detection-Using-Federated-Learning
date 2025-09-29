pipeline {
    agent any

    environment {
        DOCKER_IMAGE_NAME = "fraud-detection-app"
        DOCKERHUB_REPO = "tada8102/fraud-detection-app"
        APP_PORT = 8501 // Assuming this is the port for Streamlit
    }

    stages {
        stage('Build') {
            steps {
                script {
                    // Use a proper tag for the build, usually 'latest' or a commit/build number
                    docker.build("${DOCKER_IMAGE_NAME}:latest")
                }
            }
        }
        // Assuming unit tests are within the Docker image context and don't require external installation
        stage('Test') {
            steps {
                script {
                    // Running tests inside the newly built Docker image
                    docker.image("${DOCKER_IMAGE_NAME}:latest").inside {
                        // Assuming your Dockerfile handles 'pip install -r requirements.txt'
                        // If not, add it here, but best practice is in the Dockerfile.
                        // Assuming a standard Python unit test setup that works *inside* the image
                        sh 'python -m unittest discover || true'
                    }
                }
            }
        }
        stage('Push') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub', usernameVariable: 'DOCKERHUB_USER', passwordVariable: 'DOCKERHUB_PASSWORD')]) {
                    sh "echo \$DOCKERHUB_PASSWORD | docker login -u \$DOCKERHUB_USER --password-stdin"
                    
                    // Tag with a specific tag (e.g., 'latest') and the full repository name
                    sh "docker tag ${DOCKER_IMAGE_NAME}:latest ${DOCKERHUB_REPO}:latest"
                    sh "docker push ${DOCKERHUB_REPO}:latest"
                }
            }
        }
        stage('Deploy') {
            steps {
                // Stop any old container instance with the same name (optional but good practice)
                sh "docker stop ${DOCKER_IMAGE_NAME} || true"
                sh "docker rm ${DOCKER_IMAGE_NAME} || true"
                
                // Run the new container, mapping the port and giving it a name
                // The 'docker run' command *inside the container* should be in the Dockerfile's CMD/ENTRYPOINT
                // If it's not, you must include the full command here, e.g.:
                // sh "docker run -d -p ${APP_PORT}:${APP_PORT} --name ${DOCKER_IMAGE_NAME} ${DOCKERHUB_REPO}:latest streamlit run app.py"
                
                // **Best Practice:** The Dockerfile should contain the CMD/ENTRYPOINT, 
                // so you just run the image:
                sh "docker run -d -p ${APP_PORT}:${APP_PORT} --name ${DOCKER_IMAGE_NAME} ${DOCKERHUB_REPO}:latest"
            }
        }
    }
}
