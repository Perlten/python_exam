import bs4
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.firefox.options import Options
import time

def get_info(fruit):
    start = time.time()
    start1 = time.time()
    
    base_url = 'https://www.nemlig.com/'
    browser = webdriver.Firefox()

    # options = Options()
    # options.add_argument('--headless')
    # browser = webdriver.Firefox(options=options)

    browser.get(base_url)

    search_field = browser.find_element_by_tag_name('input')
    search_field.send_keys(fruit)
    search_field.submit()

    select = Select(browser.find_element_by_id('filter-sorting'))
    browser.implicitly_wait(1)
    # select by visible text
    select.select_by_visible_text('Billigst')

    # browser.find_element_by_xpath("//a[@class='productlist-item__link']").click()
    # sleep(1)

    #Fetch the HTML and close the browser
    page_source = browser.page_source
    browser.quit()
    end1 = time.time()
    print(end1 - start1)

    start2 = time.time()
    soup = bs4.BeautifulSoup(page_source, 'html.parser')

    #Find all tags containing the wanted values. Tag name is equal to the value to the right.
    price_cells = soup.find_all('div', {'class':'pricecontainer__base-price'})
    name_cells = soup.find_all('div', {'data-automation':'nmItemOnPg'})
    product_links = soup.find_all('a', {'class':'productlist-item__link'})
    
    products = []
    counter = 0
    #Loop through the name cells to match the first 4 to the wanted fruit
    for i, product in enumerate(name_cells):
        name = product.getText()
        if name.startswith(fruit) and counter < 4:
            counter += 1
            #Take the values out of the given tags
            price = price_cells[i].select("span")[0].getText()
            decimals = price_cells[i].select("sup")[0].getText()
            link = base_url + product_links[i]['href']
            #Convert the price to one float, as it comes in two values.
            #Add it all to a list as tuples
            products.append((float(f"{price}.{decimals}"), name, link))
            # print(f"{price}.{decimals} - {name} - {link}")
    
    for x in products:
        print(x)
    end2 = time.time()
    end = time.time()
    print(end2 - start2)
    print(end - start)
    return products

if __name__ == "__main__":
    get_info('Banan')