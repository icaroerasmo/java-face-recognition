package com.icaroerasmo.properties;

import lombok.Data;

@Data
public class CameraProperties {
    private String name;
    private String url;
    private TransportProtocol protocol = TransportProtocol.TCP;

    public enum TransportProtocol {
        TCP,
        UDP;
    }
}
