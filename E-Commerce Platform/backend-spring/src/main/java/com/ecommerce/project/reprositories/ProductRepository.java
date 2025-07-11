package com.ecommerce.project.reprositories;

import com.ecommerce.project.model.Category;
import com.ecommerce.project.model.Product;
import java.util.List;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.domain.Specification;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.JpaSpecificationExecutor;
import org.springframework.stereotype.Repository;


@Repository
public interface ProductRepository extends JpaRepository<Product, Long>,
    JpaSpecificationExecutor<Product> {

  Page<Product> findByCategoryOrderByPriceAsc(Category category, Pageable pageDetail);

  Page<Product> findByProductNameLikeIgnoreCase(String productName, Pageable pageDetail);

}
