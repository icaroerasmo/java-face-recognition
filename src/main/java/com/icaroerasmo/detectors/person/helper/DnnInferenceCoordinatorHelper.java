package com.icaroerasmo.detectors.person.helper;

import lombok.extern.log4j.Log4j2;
import org.springframework.stereotype.Component;

import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Supplier;

@Log4j2
@Component
public class DnnInferenceCoordinatorHelper {

    private final ReentrantLock inferenceLock = new ReentrantLock(true);

    public <T> T runExclusive(String modelName, Supplier<T> action) {
        inferenceLock.lock();
        try {
            return action.get();
        } finally {
            inferenceLock.unlock();
        }
    }

    public void runExclusive(String modelName, Runnable action) {
        runExclusive(modelName, () -> {
            action.run();
            return null;
        });
    }
}
