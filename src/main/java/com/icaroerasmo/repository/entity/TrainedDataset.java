package com.icaroerasmo.repository.entity;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

@Entity
@Table(name = "trained_dataset")
@Getter
@Setter
public class TrainedDataset {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private Long id;

    @Lob
    @Column(name = "model_xml", nullable = false)
    private byte[] modelXml;
}

