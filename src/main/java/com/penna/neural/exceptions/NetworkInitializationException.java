package com.penna.neural.exceptions;

@SuppressWarnings("serial")
public class NetworkInitializationException extends Exception {

    public NetworkInitializationException() {
        super();
    }
    
    public NetworkInitializationException(String msg) {
        super(msg);
    }
}
