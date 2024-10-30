plugins {
    kotlin("jvm") version "1.9.21"
    kotlin("plugin.serialization") version "1.9.22"
    application
}

group = "me.alex_s168"
version = "1.0-SNAPSHOT"

application {
    mainClass = "me.alex_s168.uiua.MainKt"
}

repositories {
    mavenCentral()
    maven {
        name = "alex's repo"
        url = uri("http://207.180.202.42:8080/libs")
        isAllowInsecureProtocol = true
    }
}

dependencies {
    testImplementation("org.jetbrains.kotlin:kotlin-test")
    implementation("org.jetbrains.kotlinx:kotlinx-io-core:0.3.2")
    implementation("me.alex_s168:blitz:0.23")
    implementation(kotlin("reflect"))
}

tasks.test {
    useJUnitPlatform()
}
kotlin {
    jvmToolchain(17)
}