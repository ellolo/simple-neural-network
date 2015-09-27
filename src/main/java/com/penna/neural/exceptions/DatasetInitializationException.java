package com.penna.neural.exceptions;

@SuppressWarnings("serial")
public class DatasetInitializationException extends Exception {

    public DatasetInitializationException() {
        super();
    }

    public DatasetInitializationException(String msg) {
        super(msg);
    }
}
