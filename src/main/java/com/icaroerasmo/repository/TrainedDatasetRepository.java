package com.icaroerasmo.repository;

import com.icaroerasmo.repository.entity.TrainedDataset;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface TrainedDatasetRepository extends JpaRepository<TrainedDataset, Long> {
}

