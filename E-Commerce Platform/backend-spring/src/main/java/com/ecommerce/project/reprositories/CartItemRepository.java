package com.ecommerce.project.reprositories;

import com.ecommerce.project.model.Cart;
import com.ecommerce.project.model.CartItem;
import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

@Repository
public interface CartItemRepository extends JpaRepository<CartItem, Long> {

  @Query("SELECT ci FROM CartItem ci WHERE ci.cart.cartId = ?1 AND ci.product.productId = ?2")
  CartItem findCartItemByCartIdAndProductId(Long cartId, Long productId);

  @Modifying
  @Query("DELETE FROM CartItem ci where ci.cart.cartId = ?1 AND ci.product.productId= ?2")
  void deleteCartItemByCartIdAndProductId(Long cartId, Long productId);


  @Modifying
  @Query("DELETE FROM CartItem ci WHERE ci.cart.cartId =?1")
  void deleteAllByCartId(Long cartId);

}
