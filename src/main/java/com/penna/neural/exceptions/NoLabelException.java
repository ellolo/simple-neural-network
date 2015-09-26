package com.penna.neural.exceptions;

@SuppressWarnings("serial")
public class NoLabelException extends Exception {

    public NoLabelException() {
        super();
    }

    public NoLabelException(String msg) {
        super(msg);
    }
}
