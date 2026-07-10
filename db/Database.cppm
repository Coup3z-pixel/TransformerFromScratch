namespace VectorDB {
  class Database {
  public:
    Database() = default;
    Database(Database &&) = default;
    Database(const Database &) = default;
    Database &operator=(Database &&) = default;
    Database &operator=(const Database &) = default;
    ~Database() = default;

    /** 
     * @param token
     * @param vector
     * @return if the vector exists already
     */
    bool insert_vector(std::string token, std::vector<double>* vector) {
      return false;
    }

    std::vector<double>* select_vector(std::string token) {
      return nullptr;
    }
  
  private:
    
  };
  
}
