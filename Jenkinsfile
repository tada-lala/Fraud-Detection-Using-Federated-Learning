pipeline {
    agent any

    environment {
        DOCKER_IMAGE = "fraud-detection-app"
        DOCKERHUB_REPO = "tada8102/fraud-detection-app"
    }

    stages {
        stage('Build') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub', usernameVariable: 'DOCKERHUB_USER', passwordVariable: 'DOCKERHUB_PASS')]) {
                    sh "echo $DOCKERHUB_PASS | docker login -u $DOCKERHUB_USER --password-stdin"
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
                        sh 'python -m unittest discover'
                    }
                }
            }
        }
        stage('Push') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub-credentials', usernameVariable: 'DOCKERHUB_USER', passwordVariable: 'DOCKERHUB_PASS')]) {
                    sh "echo $DOCKERHUB_PASS | docker login -u $DOCKERHUB_USER --password-stdin"
                    sh "docker tag $DOCKER_IMAGE $DOCKERHUB_REPO"
                    sh "docker push $DOCKERHUB_REPO"
                }
            }
        }
    }
}
