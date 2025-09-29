pipeline {
    agent any

    environment {
        DOCKER_IMAGE = "fraud-detection-app"
        DOCKERHUB_REPO = "yourdockerhubusername/fraud-detection-app"
    }

    stages {
        stage('Build') {
            steps {
                script {
                    sh 'docker build -t $DOCKER_IMAGE .'
                }
            }
        }

        stage('Test') {
            steps {
                script {
                    sh 'pip install --upgrade pip'
                    sh 'pip install -r requirements.txt'
                    sh 'pytest --maxfail=1 --disable-warnings'
                }
            }
        }

        stage('Push') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                    sh 'docker login -u $DOCKER_USER -p $DOCKER_PASS'
                    sh 'docker tag $DOCKER_IMAGE $DOCKERHUB_REPO:latest'
                    sh 'docker push $DOCKERHUB_REPO:latest'
                }
            }
        }

        stage('Deploy') {
            steps {
                echo 'Deploy your Docker container to server or Kubernetes here'
            }
        }
    }
}
