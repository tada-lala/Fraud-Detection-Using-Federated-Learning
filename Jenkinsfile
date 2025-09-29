pipeline {
    agent any

    environment {
        DOCKER_IMAGE = "fraud-detection-app"
        DOCKERHUB_REPO = "tada8102/fraud-detection-app"
    }

    stages {
        stage('Build') {
            steps {
                    withCredentials([usernamePassword(credentialsId: 'dockerhub', usernameVariable: 'DOCKERHUB_USER', passwordVariable: 'DOCKERHUB_PASSWORD')]) {
                        sh "echo $DOCKERHUB_PASSWORD | docker login -u $DOCKERHUB_USER --password-stdin"
                        script {
                            docker.build(DOCKER_IMAGE)
                        }
                    }
            }
        }
        stage('Test') {
            steps {
                script {
                    docker.image(DOCKER_IMAGE).inside {
                        sh 'pip install --no-cache-dir -r requirements.txt'
                        sh 'python -m unittest discover || true'
                    }
                }
            }
        }
        stage('Push') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub', usernameVariable: 'DOCKERHUB_USER', passwordVariable: 'DOCKERHUB_PASSWORD')]) {
                    sh "echo $DOCKERHUB_PASSWORD | docker login -u $DOCKERHUB_USER --password-stdin"
                    sh "docker tag $DOCKER_IMAGE $DOCKERHUB_REPO"
                    sh "docker push $DOCKERHUB_REPO"
                }
            }
        }
        stage('Deploy') {
            steps {
                sh "docker run -d -p 8501:8501 $DOCKERHUB_REPO streamlit run app.py"
            }
        }
    }
}
