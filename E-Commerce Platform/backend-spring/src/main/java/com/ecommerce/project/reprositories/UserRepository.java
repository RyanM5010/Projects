package com.ecommerce.project.reprositories;

import com.ecommerce.project.model.User;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface UserRepository extends JpaRepository<User, Long> {
  Optional<User> findByUserName(String username);

  boolean existsByUserName(String userName);

  boolean existsByEmail(String email);
}
