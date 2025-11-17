package com.icaroerasmo.repository;

import com.icaroerasmo.repository.entity.TrainingMetadata;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.JpaSpecificationExecutor;
import org.springframework.stereotype.Repository;

@Repository
public interface TrainingMetadataRepository extends JpaRepository<TrainingMetadata, Long>, JpaSpecificationExecutor<TrainingMetadata> {
}

