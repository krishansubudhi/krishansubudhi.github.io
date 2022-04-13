---
comments: true
author: krishan
layout: post
categories: java
title: Building a simple java app with external dependencies
description: This blog describes, how we can build a simple java application using couple of external dependencies. This can help in decopuling certain components from a big corporate application and test it.
---
This blog describes, how we can build a simple java application using couple of external dependencies. This can help in decopuling certain components from a big corporate application and test it.

We will use maven to create the project and the dependency.
## Create project:
Follwing official [maven instructions](https://maven.apache.org/guides/getting-started/maven-in-five-minutes.html), let's create a project

    mvn archetype:generate -DgroupId=com.mycompany.app -DartifactId=my-app -DarchetypeArtifactId=maven-archetype-quickstart -DarchetypeVersion=1.4 -DinteractiveMode=false

## Dependency
We will take dependency on couple of logging realted jars.
## Write code
```java
import lombok.extern.log4j.Log4j2;
@Log4j2
public class App 
{
    public static void main( String[] args )
    {
        System.out.println( "\nHello World!"+ log.getClass() );
        log.error("\nHello world from logger log4j");
    }
}
```
## Add dependencies
Add the following 3 dependencies in `pom.xml`'s 
```
<dependencies>
...
</dependencies>
```
section. 


These dependencies should be provided both during compile and run time.

    <dependency>
        <groupId>org.projectlombok</groupId>
        <artifactId>lombok</artifactId>
        <version>1.18.22</version>
        <scope>provided</scope>
    </dependency>
    <dependency>
        <groupId>org.apache.logging.log4j</groupId>
        <artifactId>log4j-api</artifactId>
        <version>2.17.2</version>
    </dependency>
    <dependency>
        <groupId>org.apache.logging.log4j</groupId>
        <artifactId>log4j-core</artifactId>
        <version>2.17.2</version>
    </dependency>

## Build and run
This shortcut command both creates and runs the final application by properly resolving dependencies.

    mvn exec:java -Dexec.mainClass="com.mycompany.app.App"'

This should print couple of messages. One from `System.out.println` and another from `log.error`.

If you are using `mvn package` just to build and planning to run the jar after that, then make sure to provide the dependencies again while running the final jar or create a jar with dependencies in the first place.
